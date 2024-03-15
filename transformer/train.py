import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data import LangDataset
from model import TranslateFormer, Tokens
from tokenizers import Tokenizer
import json
import time

lang1, lang2 = "en", "it"

with open("config.json", "r") as f:
    config = json.loads(f.read())


train_ds = LangDataset(lang1=lang1, lang2=lang2, split="train")
test_ds = LangDataset(lang1=lang1, lang2=lang2, split="test")

train_dl = DataLoader(dataset=train_ds,
                      batch_size=config["batch_size"],
                      shuffle=True)
test_dl = DataLoader(dataset=test_ds,
                     batch_size=config["batch_size"])


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
config["device"] = device
config["input_vocab_size"] = train_ds.lang1_tokenizer.get_vocab_size()
config["output_vocab_size"] = train_ds.lang2_tokenizer.get_vocab_size()
config["seq_len"] = train_ds.max_seq_len

model = TranslateFormer(config=config).to(device)
optimizer = torch.optim.AdamW(params=model.parameters(), lr=3e-4)
loss_fn = nn.CrossEntropyLoss(ignore_index=Tokens.pad_token)


def train_step() -> float:
    model.train()
    train_loss = 0
    for batch_idx, batch in enumerate(train_dl):

        encoder_input: torch.Tensor = batch["encoder_input"].to(config["device"])
        decoder_input: torch.Tensor = batch["decoder_input"].to(config["device"])
        label: torch.Tensor = batch["label"].to(config["device"])
        encoder_mask: torch.Tensor = batch["encoder_mask"].to(config["device"])
        decoder_mask: torch.Tensor = batch["decoder_mask"].to(config["device"])

        B, SEQ_LEN = label.shape
        logits: torch.Tensor = model(encoder_input, decoder_input, encoder_mask, decoder_mask) # B, SEQ_LEN, OUTPUT_VOCAB_SIZE
        loss = loss_fn(logits.view((B*SEQ_LEN, -1)), label.view((B*SEQ_LEN,)))
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    
    return train_loss/len(train_dl)

def eval_step() -> float:
    model.eval()
    eval_loss = 0
    for batch_idx, batch in enumerate(test_dl):

        encoder_input: torch.Tensor = batch["encoder_input"].to(config["device"])
        decoder_input: torch.Tensor = batch["decoder_input"].to(config["device"])
        label: torch.Tensor = batch["label"].to(config["device"])
        encoder_mask: torch.Tensor = batch["encoder_mask"].to(config["device"])
        decoder_mask: torch.Tensor = batch["decoder_mask"].to(config["device"])

        B, SEQ_LEN = label.shape

        logits: torch.Tensor = model(encoder_input, decoder_input, encoder_mask, decoder_mask) # B, SEQ_LEN, OUTPUT_VOCAB_SIZE

        loss = loss_fn(logits.view((B*SEQ_LEN, -1)), label.view((B*SEQ_LEN,)))
        eval_loss += loss.item()
    
    return eval_loss/len(test_dl)



def train():
    start = time.time()
    for epoch in range(1, config["epochs"]+1):
        train_loss = train_step()
        eval_loss = eval_step()
        print(f"({epoch}/{config['epochs']}) train: {train_loss:.4f} eval: {eval_loss:.4f} ({time.time()-start:.2f}s)")


if __name__ == "__main__":
    params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f"Params: {params/1000000:.3f}M")
    train()