from data.data import LangDataset
from tokenizers import Tokenizer
import json
from models.transformer import TranslateFormer, Tokens
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from contextlib import nullcontext
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

lang1, lang2 = "en", "fr"

tokenizer = Tokenizer.from_file(f"tokenizer/{lang1}-{lang2}.json")

with open("models/config.json", "r") as f:
    config = json.loads(f.read())

train_ds = LangDataset(lang1, lang2, "train")
test_ds = LangDataset(lang1, lang2, "test")

config["vocab_size"] = tokenizer.get_vocab_size()
config["max_seq_len"] = max(train_ds.max_seq_len, test_ds.max_seq_len)
config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

ctx = autocast(enabled=True, dtype=torch.float16) if torch.cuda.is_available() else nullcontext()
scaler = GradScaler(enabled=True if torch.cuda.is_available() else False)
train_steps = 2000
eval_step = 100
eval_steps = 200

model = TranslateFormer(config=config).to(config["device"])
optimizer = AdamW(params=model.parameters(), lr=3e-4)

def save_checkpoint(state, filename="checkpoint.pth"):
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
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = f'runs/translate_former_experiment_{current_time}'
    writer = SummaryWriter(log_dir)

    best_eval_loss = float("inf")

    model.train()
    for step in range(1, train_steps+1):

        batch = get_batch()
        x, y = batch[:, :-1].to(config["device"]), batch[:, 1:].to(config['device'])
        with ctx:
            _, loss = model(x, y)
        
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

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
                }, filename=f'checkpoint_{current_time}.pt')


            writer.add_scalar('Loss/train', train_loss, step)
            writer.add_scalar('Loss/eval', eval_loss, step)

            print(f"({step}/{train_steps}) train: {train_loss:.4f} eval: {eval_loss:.4f}")
            train_loss = 0
    
    writer.close()

if __name__ == "__main__":
    params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f"Params: {params/1000000:.3f}M")
    train()