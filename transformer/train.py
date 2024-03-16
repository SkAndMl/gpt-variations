import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from data import LangDataset
from model import TranslateFormer, Tokens
import json
import time
from contextlib import nullcontext
from typing import Dict
import random

lang1, lang2 = "en", "it"

with open("config.json", "r") as f:
    config = json.loads(f.read())


train_ds = LangDataset(lang1=lang1, lang2=lang2, split="train")
test_ds = LangDataset(lang1=lang1, lang2=lang2, split="test")

device = "cuda" if torch.cuda.is_available() else "cpu"
ctx = autocast(enabled=True, dtype=torch.float16) if device=="cuda" else nullcontext()
scaler = GradScaler(enabled=True if device=="cuda" else False)
config["device"] = device
config["input_vocab_size"] = train_ds.lang1_tokenizer.get_vocab_size()
config["output_vocab_size"] = train_ds.lang2_tokenizer.get_vocab_size()
config["lang1_tokenizer"] = train_ds.lang1_tokenizer
config["lang2_tokenizer"] = train_ds.lang2_tokenizer
config["seq_len"] = train_ds.max_seq_len

model = TranslateFormer(config=config).to(device)
optimizer = torch.optim.AdamW(params=model.parameters(), lr=3e-4)
loss_fn = nn.CrossEntropyLoss(ignore_index=Tokens.pad_token)

def get_batch(split:str="train") -> Dict[str, torch.Tensor]:
    data = train_ds if split=="train" else test_ds
    idxs = torch.randint(low=0, high=len(data)-1, size=(config["batch_size"],))
    batch = {
        "encoder_input" : [],
        "decoder_input" : [],
        "label" : [],
        "encoder_mask" : [],
        "decoder_mask" : []
    }
    for idx in idxs:
        batch["encoder_input"].append(data[idx.item()]["encoder_input"])
        batch["decoder_input"].append(data[idx.item()]["decoder_input"])
        batch["label"].append(data[idx.item()]["label"])
        batch["encoder_mask"].append(data[idx.item()]["encoder_mask"])
        batch["decoder_mask"].append(data[idx.item()]["decoder_mask"])

    batch["encoder_input"] = torch.stack(batch["encoder_input"], dim=0).to(config["device"])
    batch["decoder_input"] = torch.stack(batch["decoder_input"], dim=0).to(config["device"])
    batch["label"] = torch.stack(batch["label"], dim=0).to(config["device"])
    batch["encoder_mask"] = torch.stack(batch["encoder_mask"], dim=0).to(config["device"])
    batch["decoder_mask"] = torch.stack(batch["decoder_mask"], dim=0).to(config["device"])

    return batch


@torch.inference_mode()
def eval_step() -> float:
    model.eval()
    eval_loss = 0
    for step in range(config["eval_steps"]):

        batch = get_batch("test")

        with ctx:
            logits = model(batch["encoder_input"],
                           batch["decoder_input"],
                           batch["encoder_mask"],
                           batch["decoder_mask"])
            B, SEQ_LEN, _ = logits.shape
            loss = loss_fn(logits.view((B*SEQ_LEN, -1)), batch["label"].view((B*SEQ_LEN,)))
        
        eval_loss += loss.item()
    
    return eval_loss/config["eval_steps"]


def train():
    start = time.time()
    train_loss = 0
    for step in range(1, config["train_steps"]+1):

        batch = get_batch(split="train")
        with ctx:
            logits = model(batch["encoder_input"],
                           batch["decoder_input"],
                           batch["encoder_mask"],
                           batch["decoder_mask"])
            B, SEQ_LEN, _ = logits.shape
            loss = loss_fn(logits.view((B*SEQ_LEN, -1)), batch["label"].view((B*SEQ_LEN,)))
        
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

        if step%config["eval_step"]==0:
            eval_loss = eval_step()
            train_loss = train_loss/config["eval_step"]
            print(f"{step}/{config['train_steps']} train: {train_loss:.4f} eval: {eval_loss:.4f} ({time.time()-start:.2f}s)")
            
            rand_idx = random.randint(0, len(test_ds)-1)
            sample = test_ds[rand_idx]
            input_sentence, output_sentence = test_ds.ds["test"][rand_idx]["translation"][lang1], \
                                              test_ds.ds["test"][rand_idx]["translation"][lang2]
            
            translation: str = model.translate(input_tokens=sample["encoder_input"].to(device),
                                               encoder_mask=sample["encoder_mask"].to(device))
            print(f"<= {input_sentence}")
            print(f"== {output_sentence}")
            print(f"=> {translation}")
            print("-"*100)

            train_loss = 0
            model.train()


if __name__ == "__main__":
    params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f"Params: {params/1000000:.3f}M")
    train()