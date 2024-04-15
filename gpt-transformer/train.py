import torch
from typing import Tuple, Dict
import json
from torch.cuda.amp import GradScaler, autocast
from contextlib import nullcontext
from torch.utils.tensorboard.writer import SummaryWriter
import os

from model import VanillaGPT, ParallelGPT, ConvGPT

device = "cuda" if torch.cuda.is_available() else "cpu"
ctx = autocast(enabled=True, dtype=torch.float16) if device=="cuda" else nullcontext()
model_name = "vanillagpt" # "parallelgpt", "convgpt"

with open("config.json", "r") as f:
    config = json.load(f)

with open("train.txt", "r", encoding="utf-8") as train_file, open("valid.txt", "r", encoding="utf-8") as valid_file:
    train_data, valid_data = train_file.read(), valid_file.read()
    data = train_data + valid_data
    train_len = len(train_data)


chars = sorted(list(set(data)))
vocab_size = len(chars)
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

encode = lambda s: [stoi[ch] for ch in s]
decode = lambda l: "".join([itos[i] for i in l])

data = torch.tensor(encode(data))

train_data = data[:train_len]
val_data = data[train_len:]

if model_name.startswith("parallel"):
    gpt = ParallelGPT(config=config, vocab_size=vocab_size)
elif model_name.startswith("conv"):
    gpt = ConvGPT(config=config, vocab_size=vocab_size)
else:
    gpt = VanillaGPT(config=config, vocab_size=vocab_size)

gpt = gpt.to(device)
optimizer = torch.optim.AdamW(params=gpt.parameters(),
                              lr=config["learning_rate"],
                              weight_decay=config["weight_decay"])
scaler = GradScaler(enabled=True)
writer = SummaryWriter(log_dir=f"runs/{model_name}")


def get_random_batch(split: str="train") -> Tuple[torch.Tensor, torch.Tensor]:

    data = train_data if split=="train" else val_data

    batch_size = config["batch_size"]
    block_size = config["block_size"]

    idxs = torch.randint(0, len(data)-block_size, size=(batch_size,))
    x_batch = torch.stack([data[i:i+block_size] for i in idxs])
    y_batch = torch.stack([data[i+1:i+block_size+1] for i in idxs])

    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
    return x_batch, y_batch


@torch.no_grad()
def eval_model() -> Dict[str, float]:
    losses = {}
    gpt.eval()

    for split in ["train", "val"]:
        loss = 0
        for _ in range(config["eval_iters"]):
            x_batch, y_batch = get_random_batch(split)
            
            with ctx:
                _, l_ = gpt(x_batch, y_batch)
            
            loss += l_.item()
        
        losses[split] = loss/config["eval_iters"]
    
    gpt.train()
    return losses


def train():
    gpt.train()
    for iter in range(1, config["train_iters"]+1):

        if iter%config["eval_interval"]==0:
            losses = eval_model()

            for k in losses:
                writer.add_scalar(tag=f"loss/{k}",
                                  scalar_value=losses[k],
                                  global_step=iter)

            print(f"iter {iter} train_loss: {losses['train']} val_loss: {losses['val']}")
        
        x_batch, y_batch = get_random_batch()
    
        with ctx:
            _, loss = gpt(x_batch, y_batch)

        scaler.scale(loss).backward()  
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    if not os.path.exists("checkpoints/"):
      os.mkdir("checkpoints")

    torch.save(gpt.state_dict(),
               f=f"checkpoints/{model_name}.pt")


if __name__ == "__main__":
    params = sum([torch.numel(p) for p in gpt.parameters() if p.requires_grad])
    print(f"Params: {params/1000000:.3}M")
    print(f"Model size: {params*4/(1024*1024):.2f} MB")
    train()