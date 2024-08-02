"""
adapted from https://github.com/karpathy/nanoGPT/blob/master/train.py
"""

import os
import time
import math
from contextlib import nullcontext

import numpy as np
import torch
from dataclasses import dataclass
from models import GPTConfig, GPT, ParallelGPT, ConvGPT, LinearGPT

# -----------------------------------------------------------------------------
# I/O
@dataclass
class TrainConfig:
    out_dir = 'out'
    eval_interval: int = 1000
    log_interval = 100
    eval_iters = 100
    warmup_iters = 2000
    max_iters = 10000
    lr_decay_iters = 10000 
    eval_only = False 
    always_save_checkpoint = True 
    dataset = 'data'
    gradient_accumulation_steps = 4
    batch_size: int = 16
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'  
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    data_dir = os.path.join('data', dataset)
    weight_decay = 1e-2
    beta1 = 0.9
    beta2 = 0.95
    learning_rate = 3e-4
    min_lr = 3e-5
    decay_lr = True
    grad_clip = 1.0    


def train(train_config: TrainConfig, model_config: GPTConfig):

    if model_config.model_type=='gpt': model = GPT(model_config)
    elif model_config.model_type=='pgpt': model=ParallelGPT(model_config)
    elif model_config.model_type=='cgpt': model=ConvGPT(model_config)
    elif model_config.model_type=='lgpt': model=LinearGPT(model_config)

    os.makedirs(train_config.out_dir, exist_ok=True)
    
    def get_batch(split):
        if split == 'train':
            data = np.memmap(os.path.join(train_config.data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        else:
            data = np.memmap(os.path.join(train_config.data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        ix = torch.randint(len(data) - model_config.block_size, (train_config.batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+model_config.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+model_config.block_size]).astype(np.int64)) for i in ix])
        if train_config.device_type == 'cuda':
            x, y = x.pin_memory().to(train_config.device, non_blocking=True), y.pin_memory().to(train_config.device, non_blocking=True)
        else:
            x, y = x.to(train_config.device), y.to(train_config.device)
        return x, y

    def get_lr(it):
        if it < train_config.warmup_iters:
            return train_config.learning_rate * it / train_config.warmup_iters
        if it > train_config.lr_decay_iters:
            return train_config.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - train_config.warmup_iters) / (train_config.lr_decay_iters - train_config.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return train_config.min_lr + coeff * (train_config.learning_rate - train_config.min_lr)

    @torch.inference_mode()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(train_config.eval_iters)
            for k in range(train_config.eval_iters):
                X, Y = get_batch(split)
                with train_config.ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    model = model.to(train_config.device)
    iter_num = 0
    best_val_loss = 1e9
    scaler = torch.cuda.amp.GradScaler(enabled=(train_config.dtype == 'float16'))
    optimizer = model.configure_optimizers(train_config.weight_decay, train_config.learning_rate, (train_config.beta1, train_config.beta2), train_config.device_type)

    X, Y = get_batch('train') 
    t0 = time.time()

    while True:

        lr = get_lr(iter_num) if train_config.decay_lr else train_config.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % train_config.eval_interval == 0:
            losses = estimate_loss()
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'model_config': model_config,
                        'train_config': train_config
                    }
                    print(f"saving checkpoint to {train_config.out_dir}")
                    torch.save(checkpoint, os.path.join(train_config.out_dir, 'ckpt.pt'))

        for micro_step in range(train_config.gradient_accumulation_steps):
            with train_config.ctx:
                logits, loss = model(X, Y)
                loss = loss / train_config.gradient_accumulation_steps # scale the loss to account for gradient accumulation
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch('train')
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
        # clip the gradient
        if train_config.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % train_config.log_interval == 0:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * train_config.gradient_accumulation_steps
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
        iter_num += 1
        # termination conditions
        if iter_num > train_config.max_iters:
            break