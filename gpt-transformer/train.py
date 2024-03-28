from data import LangDataset
from tokenizers import Tokenizer
import json
from scripts import Tokens, initialize_weights
from translate_former import PosTranslateFormer, ConvTranslateFormer
from translate_former import ParallelTranslateFormer, TranslateFormer
import torch
from torch import nn as nn
from torch.optim import Adam
from contextlib import nullcontext
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils.rnn import pad_sequence
from torchmetrics.text import BLEUScore, WordErrorRate, CharErrorRate
from torch.utils.tensorboard import SummaryWriter
import logging
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

model_name = "translateformer"
log_dir = f'tensorboard_results/{model_name}'
writer = SummaryWriter(log_dir)

with open("config.json", "r") as f:
    config = json.loads(f.read())

tokenizer: Tokenizer = Tokenizer.from_file(config['tokenizer_file_path'])

train_ds = LangDataset(config["lang1"], config["lang2"], "train")
test_ds = LangDataset(config["lang1"], config["lang2"], "test")

config["vocab_size"] = tokenizer.get_vocab_size()
config["seq_len"] = LangDataset.calc_max_seq_len(config["lang1"], config["lang2"])
config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

ctx = autocast(enabled=True, dtype=torch.float16) if torch.cuda.is_available() else nullcontext()
scaler = GradScaler(enabled=True if torch.cuda.is_available() else False)
train_steps = 40000
eval_step = 1000
eval_steps = 1000
gradient_accumulation_steps = 2

if model_name.startswith("pos"):
    model = PosTranslateFormer(config=config)
elif model_name.startswith("conv"):
    model = ConvTranslateFormer(config=config)
elif model_name.startswith("par"):
    model = ParallelTranslateFormer(config=config)
else:
    model = TranslateFormer(config=config)

model = model.to(config["device"])
model.apply(initialize_weights)
logging.info("Initialized model")
optimizer = Adam(params=model.parameters(), lr=config["initial_lr"], weight_decay=config["weight_decay"])

bleu = BLEUScore()
wer = WordErrorRate()
cer = CharErrorRate()


def log_results(train_loss: float, eval_loss: float, step: int) -> None:

    rand_idx = random.sample(range(0, len(test_ds)), k=1)[0]
    sample: str = test_ds.data[rand_idx]
    sep_idx = sample.find("<sep>")
    
    src, tgt = sample[5:sep_idx], sample[sep_idx+5:-5]
    src, tgt = src.strip(), tgt.strip()

    pred = model.translate(x=src)
    
    bleu.update([pred], [[tgt]])
    wer.update([pred], [tgt])
    cer.update([pred], [tgt])

    bleu_score: float = bleu.compute().item()
    wer_score: float = wer.compute().item()
    cer_score: float = cer.compute().item()

    bleu.reset()
    wer.reset()
    cer.reset()

    ## add losses to tensorboard
    writer.add_scalar(
        tag = "Loss/train", 
        scalar_value=train_loss,
        global_step=step
    )
    writer.add_scalar(
        tag = "Loss/test", 
        scalar_value=eval_loss,
        global_step=step
    )

    ## add metrics to tensorboard
    writer.add_scalar(
        tag="Metric/bleu",
        scalar_value=bleu_score,
        global_step=step
    )
    writer.add_scalar(
        tag="Metric/wer",
        scalar_value=wer_score,
        global_step=step
    )
    writer.add_scalar(
        tag="Metric/cer",
        scalar_value=cer_score,
        global_step=step
    )


def save_checkpoint(state, filename="checkpoint.pt"):
    torch.save(state, filename)


def get_batch(split:str="train") -> torch.Tensor:
    data = train_ds if split=="train" else test_ds
    idxs = torch.randint(low=0, high=len(data)-1, size=(config["batch_size"],))
    batch = [data[idx.item()] for idx in idxs]
    return pad_sequence(batch, batch_first=True, padding_value=Tokens.pad_token)


@torch.inference_mode()
def eval_model() -> float:
    model.eval()
    batch = get_batch(split="test")
    x, y = batch[:, :-1].to(config["device"]), batch[:, 1:].to(config["device"])
    eval_loss = 0
    for _ in range(eval_steps):
        with ctx:
            _, loss = model(x, y)
        eval_loss += loss.item()
    model.train()
    return eval_loss/eval_steps


def train():
    train_loss = 0
    best_eval_loss = float("inf")

    model.train()
    for step in range(1, train_steps+1):

        for _ in range(gradient_accumulation_steps):
            batch = get_batch()
            x, y = batch[:, :-1].to(config["device"]), batch[:, 1:].to(config['device'])
            with ctx:
                _, loss = model(x, y)
            
            loss /= gradient_accumulation_steps
            scaler.scale(loss).backward()
            train_loss += loss.item()
        
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        

        if step%eval_step==0:
            eval_loss = eval_model()
            train_loss /= eval_step

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                save_checkpoint({
                    'epoch': step // eval_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': eval_loss,
                }, filename=f'{model_name}_cp.pt')


            log_results(train_loss, eval_loss, step)

            print(f"({step*100/train_steps:.2f}%) train: {train_loss:.4f} eval: {eval_loss:.4f}")
            train_loss = 0
    
    writer.close()

if __name__ == "__main__":
    params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f"Params: {params/1000000:.3f}M")
    train()