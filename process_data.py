from datasets import load_dataset
import numpy as np
import os

ds = load_dataset('Salesforce/wikitext', 'wikitext-2-v1')
train_txt, val_txt = '', ''

for item in ds['train']: train_txt += item['text']
for item in ds['validation']: train_txt += item['text']
for item in ds['test']: val_txt += item['text']

chars = sorted(list(set(train_txt + val_txt)))
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for ch, i in enumerate(chars)}
encode_fn = lambda s: [stoi[ch] for ch in s]
print(f'vocab size: {len(chars)}')


with open('data/train.txt', 'w') as f: f.write(train_txt)
with open('data/val.txt', 'w') as f: f.write(val_txt)

train_ids = encode_fn(train_txt)
val_ids = encode_fn(val_txt)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join('data', 'train.bin'))
val_ids.tofile(os.path.join('data', 'val.bin'))