import numpy as np
import os
import json

with open('data/shakespeare.txt', 'r') as f: data = f.read()


chars = sorted(list(set(data)))
n = len(data)
train_txt = data[:int(n*0.9)]
val_txt = data[int(n*0.9):]
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode_fn = lambda s: [stoi[ch] for ch in s]
print(f'vocab size: {len(chars)}')

with open('data/stoi.json', 'w') as f: json.dump(stoi, f)
with open('data/itos.json', 'w') as f: json.dump(itos, f)
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