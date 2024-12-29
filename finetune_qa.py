from datasets import load_dataset
from evaluate import load
import bert_score
import tiktoken
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from models import GPTConfig, ParallelGPT, GPT
from tqdm import tqdm
from transformers import AutoTokenizer


tokenizer = tiktoken.get_encoding("gpt2")
ds = load_dataset("squad_v2", trust_remote_code=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 8
ignore_index = -100
pad_token_id = 50256
config = GPTConfig(vocab_size=50304)

def format_input(example):
    sample = (
        f"Use the following context to answer the given question.\n"
        f"## Context\n{example['context']}\n"
        f"## Question\n{example['question']}\n"
    )
    return sample

def filter_examples(example):
    sample = format_input(example)
    return len(tokenizer.encode(sample)) <= 1024

filtered_ds = ds.filter(filter_examples, 
                        batch_size=2000)


class SquadDataset(Dataset):

  def __init__(self, ds, tokenizer):
    self.ds = ds
    self.tokenized_texts = []

    for example in ds:
      if len(example['answers']['text'])>0:
          sample = format_input(example)
          sample += f"## Answer\n{example['answers']['text'][0]}"
          self.tokenized_texts.append(tokenizer.encode(sample))
    
  
  def __getitem__(self, idx): return self.tokenized_texts[idx]

  def __len__(self): return len(self.tokenized_texts)



class ValidationDataset(Dataset):

    def __init__(self, ds, tokenizer):

        self.ds = ds
        self.inputs = []
        self.answers = []

        for example in ds:
            if len(example['answers']['text'])>0:
                sample = format_input(example)
                self.inputs.append(tokenizer.encode(sample))
                self.answers.append(example['answers']['text'][0])
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.answers[idx]

    def __len__(self): return len(self.inputs)





train_ds = SquadDataset(ds=filtered_ds['train'], tokenizer=tokenizer)
test_ds = SquadDataset(ds=filtered_ds['validation'], tokenizer=tokenizer)
validation_ds = ValidationDataset(ds=filtered_ds['validation'], tokenizer=tokenizer)


def collate_fn(batch, device="cpu", pad_token_id=50256, ignore_index=-100):

  batch_max_length = max(len(item)+1 for item in batch)
  inputs_lst, targets_lst = [], []

  for item in batch:
    new_item = item.copy()
    new_item += [pad_token_id]

    padded = (
        new_item + [pad_token_id]*(batch_max_length-len(new_item))
    )

    inputs = torch.tensor(padded[:-1])
    targets = torch.tensor(padded[1:])

    mask = targets == pad_token_id
    indices = torch.nonzero(mask).squeeze()
    
    if indices.numel() > 1:
      targets[indices[1:]] = ignore_index
    
    inputs_lst.append(inputs)
    targets_lst.append(targets)
  
  return torch.stack(inputs_lst).to(device), torch.stack(targets_lst).to(device)


def collate_fn_validation(batch, device="cpu", pad_token_id=50256):

  batch_max_length = max(len(inputs)+1 for inputs, _ in batch)
  inputs_lst, targets_lst = [], []

  for inputs, targets in batch:
    new_item = inputs.copy()
    new_item += [pad_token_id]

    padded = (
        new_item + [pad_token_id]*(batch_max_length-len(new_item))
    )

    inputs = torch.tensor(padded[:-1])

    
    inputs_lst.append(inputs)
    targets_lst.append(targets)
  
  return torch.stack(inputs_lst).to(device), targets_lst



custom_collate_fn = partial(collate_fn, 
                            pad_token_id=pad_token_id, 
                            ignore_index=ignore_index,
                            device=device)
custom_validation_collate_fn = partial(collate_fn_validation,
                                       pad_token_id=pad_token_id,
                                       device=device)

train_dl = DataLoader(train_ds, batch_size=batch_size, collate_fn=custom_collate_fn,
                      shuffle=True, drop_last=True)
test_dl = DataLoader(test_ds, batch_size=batch_size, collate_fn=custom_collate_fn,
                     shuffle=False, drop_last=False)
validation_dl = DataLoader(validation_ds, batch_size=batch_size,
                           collate_fn=custom_validation_collate_fn, shuffle=False,
                           drop_last=False)

class GPTForQA(GPT):
   
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

    @torch.inference_mode()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            if idx[0, -1].item() == pad_token_id: break

        return idx


class ParallelGPTForQA(ParallelGPT):

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
    
    @torch.inference_mode()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            if idx[0, -1].item() == pad_token_id: break

        return idx


config = GPTConfig(vocab_size=50304)
model = ParallelGPTForQA(config=config, freeze_block="h_1")
cp = torch.load("pgpt.pt", map_location=device)
model.load_state_dict(cp['model'])
model = model.to(device)

optimizer = torch.optim.AdamW(params=model.parameters(),
                              lr=5e-5, 
                              weight_decay=1e-2)
loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

rouge_metric = load("rouge")
bert_scorer = bert_score.BERTScorer('bert-base-uncased')

def evaluate(model, dataloader):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            logits = model(inputs)
            loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    model.train()
    return avg_loss

@torch.inference_mode()
def report_metrics(model, tokenizer, max_new_tokens=128, temperature=0.1, top_k=None, model_name="gpt"):
    
    model.eval()

    rouge_scores = 0
    bert_scores = 0
    total = 0

    for batch in tqdm(validation_dl, desc="Evaluating"):
        inputs, references = batch
        inputs = inputs.to(device)

        generated_ids = model.generate(
            idx=inputs, 
            max_new_tokens=max_new_tokens,
            temperature=temperature, 
            top_k=top_k
        )

        generated_texts = [tokenizer.decode(ids.cpu().numpy().tolist()) for ids in generated_ids]
        reference_texts = references 

        
        
        rouge_result = rouge_metric.compute(predictions=generated_texts, references=reference_texts, use_stemmer=True)
        

        bert_batch_scores = bert_scorer.score(cands=generated_texts, refs=reference_texts)

        rouge_scores += rouge_result["rougeL"] 
        bert_scores += sum(bert_batch_scores[2].cpu().numpy().tolist())

        total += len(references)

    with open(f"log_qa_{model_name}.txt", "w") as f:
    
        f.write(f"rouge-l: {rouge_scores/len(validation_dl)}\n")
        f.write(f"bert_score: {bert_scores/total}\n")



## train
eval_freq = 1000
grad_accum_steps = 4
last_step = len(train_dl)-1
print(f"last_step: {last_step}")


for step, (inputs, targets) in enumerate(train_dl):
    
    bsz, seq_len = inputs.shape
    inputs, targets = inputs.to(device), targets.to(device)
    logits = model(inputs)

    loss = loss_fn(logits.view(bsz*seq_len, -1), targets.view(-1))
    loss.backward()

    if (step+1)%grad_accum_steps==0 or step==last_step:
        optimizer.step()
        optimizer.zero_grad()
    if (step+1)%grad_accum_steps or step==last_step:
        print(f"step={step} | train_loss: {loss.item():.6f}")

        
    if (step+1)%eval_freq==0:
        val_loss = evaluate(model, 
                            dataloader=test_dl)
        print(f"val loss: {val_loss:.6f}")
    
    
report_metrics(model, tokenizer, model_name="pgpt")


del model
del optimizer
torch.cuda.empty_cache()

model = GPTForQA(config=config)
cp = torch.load("gpt.pt", map_location=device)
model.load_state_dict(cp['model'])
model = model.to(device)

optimizer = torch.optim.AdamW(params=model.parameters(),
                              lr=5e-5, 
                              weight_decay=1e-2)

print("finetuning gpt")
## train
eval_freq = 1000
for step, (inputs, targets) in enumerate(train_dl):
    bsz, seq_len = inputs.shape
    inputs, targets = inputs.to(device), targets.to(device)
    logits = model(inputs)

    loss = loss_fn(logits.view(bsz*seq_len, -1), targets.view(-1))
    loss.backward()

    if (step+1)%grad_accum_steps==0 or step==last_step:
        optimizer.step()
        optimizer.zero_grad()
    if (step+1)%grad_accum_steps or step==last_step:
        print(f"step={step} | train_loss: {loss.item():.6f}")

        
    if (step+1)%eval_freq==0:
        val_loss = evaluate(model, 
                            dataloader=test_dl,
                           model_name="gpt")
        print(f"val loss: {val_loss:.6f}")
    
report_metrics(model, tokenizer)