from datasets import load_dataset
import tiktoken
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from models import GPTConfig, ParallelGPT, GPT
from tqdm import tqdm


tokenizer = tiktoken.get_encoding("gpt2")
ds = load_dataset("sst2", trust_remote_code=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 8
grad_accum_steps = 4
pad_token_id = 50256
config = GPTConfig(vocab_size=50304)

def format_example(example):
    sample = (
       f"Classify the following sentence as positive or negative.\n"\
       f"Sentence: {example['sentence']}"
    )
    return sample

def filter_examples(example):
    sample = format_example(example)
    return len(tokenizer.encode(sample)) <= 1023

filtered_ds = ds.filter(filter_examples, 
                        batch_size=2000)


class ClsDataset(Dataset):

  def __init__(self, ds, tokenizer):
    self.ds = ds
    self.tokenized_texts = []
    self.labels = []

    for example in ds:
      sample = format_example(example)
      self.tokenized_texts.append(tokenizer.encode(sample))
      self.labels.append(example['label'])
  
  def __getitem__(self, idx): 
     return self.tokenized_texts[idx], self.labels[idx]

  def __len__(self): return len(self.tokenized_texts)


train_ds = ClsDataset(ds=filtered_ds['train'], tokenizer=tokenizer)
test_ds = ClsDataset(ds=filtered_ds['validation'], tokenizer=tokenizer)

def collate_fn(batch, device="cpu", pad_token_id=50256):

  batch_max_length = max(len(item)+1 for item, _ in batch)
  inputs_lst, targets_lst = [], []

  for input, label in batch:
    new_item = input.copy()
    new_item += [pad_token_id]

    padded = (
        new_item + [pad_token_id]*(batch_max_length-len(new_item))
    )

    input = torch.tensor(padded)

    inputs_lst.append(input)
    targets_lst.append(label)

  # print(targets_lst)
  
  return torch.stack(inputs_lst).to(device), torch.tensor(targets_lst, dtype=torch.long).to(device)


custom_collate_fn = partial(collate_fn, 
                            pad_token_id=pad_token_id, 
                            device=device)

train_dl = DataLoader(train_ds, batch_size=batch_size, collate_fn=custom_collate_fn,
                      shuffle=True, drop_last=True)
test_dl = DataLoader(test_ds, batch_size=batch_size, collate_fn=custom_collate_fn,
                     shuffle=False, drop_last=False)


class GPTForCls(GPT):
   
    def __init__(self, config):
        super().__init__(config)    

    def forward(self, x: torch.Tensor):
        device = x.device
        _, t = x.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        tok_emb = self.transformer.wte(x) 
        pos_emb = self.transformer.wpe(pos) 
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits


class ParallelGPTForCls(ParallelGPT):

    def __init__(self, config, freeze_block="h_1"):
        super().__init__(config)

        if freeze_block=="h_1":
            self.transformer.lin_1.requires_grad = False
            for params in self.transformer.h_1.parameters():
                params.requires_grad = False
        else:
           self.transformer.lin_2.requires_grad = False
           for params in self.transformer.h_2.parameters():
              params.requires_grad = False
    
    def forward(self, x: torch.Tensor):
        
        _, seq_len = x.shape
        pos = torch.arange(0, seq_len, dtype=torch.long, device=x.device) # shape (t)

        tok_emb = self.transformer.wte(x) 
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(pos_emb + tok_emb)
        x_1 = self.transformer.lin_1(x)
        x_2 = self.transformer.lin_2(x)
        for block_1, block_2 in zip(self.transformer.h_1, self.transformer.h_2):
            x_1, x_2 = block_1(x_1), block_2(x_2)
        wt = F.sigmoid(self.transformer.wt)   
        x = wt*x_1 + (1-wt)*x_2
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)

        return logits
    

config = GPTConfig(vocab_size=50304)
model = ParallelGPTForCls(config=config, freeze_block="h_1")
cp = torch.load("pgpt.pt", map_location=device)
model.load_state_dict(cp['model'])
model.lm_head = nn.Linear(
   in_features=config.n_embd,
   out_features=len(filtered_ds['train'].features['label'].names)
)
model = model.to(device)
optimizer = torch.optim.AdamW(params=model.parameters(),
                              lr=5e-5, 
                              weight_decay=1e-2)
loss_fn = nn.CrossEntropyLoss()


def evaluate(model, dataloader):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            # print(targets)
            logits = model(inputs)[:, -1, :]
            loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    model.train()
    return avg_loss


@torch.inference_mode()
def report_metrics(model, tokenizer, model_name):
   
    model.eval()

    correct = 0
    total = 0
    for example in filtered_ds['validation']:
        input = format_example(example)
        input_ids = torch.tensor([tokenizer.encode(input)], device=device)
        logits = model(input_ids)[:, -1, :]
        pred_id = torch.argmax(logits, dim=-1)[0].item()
        correct += 1 if pred_id==example['label'] else 0
        total += 1

    with open(f"cls_{model_name}.txt", "w") as f:
        f.write(f"accuracy: {correct/total}")

print("training pgpt")

## train
eval_freq = 2000
last_step = len(train_dl)-1
for step, (inputs, targets) in enumerate(train_dl):
    bsz, _ = inputs.shape
    inputs, targets = inputs.to(device), targets.to(device)
    logits = model(inputs)[:, -1, :]

    loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
    loss.backward()

    if (step+1)%grad_accum_steps==0 or step==last_step:
        optimizer.step()
        optimizer.zero_grad()
        print(f"step={step} | train_loss: {loss.item():.6f}")

        
    if (step+1)%eval_freq==0 or step==last_step:
        val_loss = evaluate(model, 
                            dataloader=test_dl)
        print(f"val loss: {val_loss:.6f}")
    

report_metrics(model, tokenizer, model_name="pgpt")


del model
del optimizer
torch.cuda.empty_cache()


model = GPTForCls(config=config)
cp = torch.load("gpt.pt", map_location=device)
model.load_state_dict(cp['model'])
model.lm_head = nn.Linear(
   in_features=config.n_embd,
   out_features=len(filtered_ds['train'].features['label'].names)
)
model = model.to(device)
optimizer = torch.optim.AdamW(params=model.parameters(),
                              lr=5e-5, 
                              weight_decay=1e-2)


print("training gpt")

## train
eval_freq = 1000
last_step = len(train_dl)-1
for step, (inputs, targets) in enumerate(train_dl):
    bsz, _ = inputs.shape
    inputs, targets = inputs.to(device), targets.to(device)
    logits = model(inputs)[:, -1, :]

    loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
    loss.backward()

    if (step+1)%grad_accum_steps==0 or step==last_step:
        optimizer.step()
        optimizer.zero_grad()
        print(f"step={step} | train_loss: {loss.item():.6f}")

        
    if (step+1)%eval_freq==0 or step==last_step:
        val_loss = evaluate(model, 
                            dataloader=test_dl)
        print(f"val loss: {val_loss:.6f}")
    
report_metrics(model, tokenizer, model_name="gpt")