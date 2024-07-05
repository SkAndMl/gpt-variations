"""
script for training, checkpointing, inferencing, etc
June 30, 2024
"""

import matplotlib.pyplot as plt
import torch
from torch import nn
import time
import math
import random
import pandas as pd
import numpy as np
from torch.optim import AdamW
from gpts.data import TextDataset
from gpts.models import GPT, pGPT, ccGPT, lcGPT

class Trainer:

    def __init__(self, model_config, train_config):
        
        self.train_config = train_config
        self.model_config = model_config
        self.device = train_config["device"]
        self._init_model(model_config['model_type'])
        self.train_data = np.load("data/train.npy")
        self.test_data = np.load("data/test.npy")
        self.test_ds = TextDataset(train=False)
        self.ctx_length = model_config['ctx_length']
        self.train_iters = train_config['train_iters']
        self.eval_step = train_config['eval_step']
        self.eval_iters = train_config['eval_iters']
        self.accum_steps = train_config['gradient_accumulation_steps']
        self.metrics = pd.DataFrame(columns=["iter", "train_loss", "eval_loss", "lr", "time"])
        
        ## setup lr schedule params
        self.max_lr = train_config['max_lr']
        self.min_lr = train_config['min_lr']
        self.warmup_steps = train_config['warmup_steps']
        self.grad_clip = train_config.get("grad_clip", None)
        

    def _init_model(self, model):

        if model == "gpt":
            self.model = GPT(self.model_config)
        elif model == "pgpt":
            self.model = pGPT(self.model_config)
        elif model == "ccgpt":
            self.model = ccGPT(self.model_config)
        else:
            self.model = lcGPT(self.model_config)
        
        params = sum(torch.numel(p) for p in self.model.parameters() if p.requires_grad)
        print(f"Model params: {params/1e6:.3f}")
        
        self.model = self.model.to(self.device)
        self.optimizer = AdamW(params=self.model.parameters(),
                               lr=self.train_config['max_lr'],
                              weight_decay=self.train_config['wd'])

 
    def _get_batch(self, train=True):
        
        if train:
            idxs = random.sample(list(range(self.train_data.shape[0]-self.ctx_length-1)), k=self.train_config['bs'])
            blocks = [torch.tensor(self.train_data[idx:idx+self.ctx_length+1], dtype=torch.long) for idx in idxs]
            blocks = torch.stack(blocks, dim=0)
            return blocks[:, :-1], blocks[:, 1:]
        else:
            idxs = random.sample(list(range(self.test_data.shape[0]-self.ctx_length-1)), k=self.train_config['bs'])
            blocks = [torch.tensor(self.test_data[idx:idx+self.ctx_length+1], dtype=torch.long) for idx in idxs]
            blocks = torch.stack(blocks, dim=0)
            return blocks[:, :-1], blocks[:, 1:]
    
        
    def _get_lr(self, itr):
        # warmup the lr
        if itr<self.warmup_steps:
            return self.max_lr*itr/self.warmup_steps
        # return min_lr if training is continued after predefined iters
        if itr>self.train_iters:
            return self.min_lr
        
        ## cosine decay
        decay_ratio = (itr-self.warmup_steps)/(self.train_iters-self.warmup_steps)
        lr = self.min_lr + 0.5*(self.max_lr-self.min_lr)*(1.0 + math.cos(math.pi*decay_ratio))
        return lr

    def fit(self):

        running_loss = 0
        a = time.time()
        for itr in range(1, self.train_iters+1):
            self.optimizer.zero_grad()
            for accum_itr in range(self.accum_steps):
                
                x, y = self._get_batch(True)
                x, y = x.to(self.device), y.to(self.device)
                with torch.autocast(device_type=self.device, dtype=torch.float16):
                    _, loss = self.model(x, y)
                
                running_loss += loss.item()
                loss /= self.accum_steps # normalize loss according to grad accum steps
                loss.backward()
            
            # clip gradients to handle exploding gradients
            if self.grad_clip is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            lr = self._get_lr(itr) # get updated lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr # update lr
            
            self.optimizer.step()

            if itr%self.eval_step==0:
                self.model.eval()
                eval_loss = 0
                with torch.inference_mode():
                    for _ in range(self.eval_iters):
                        x, y = self._get_batch(False)
                        x, y = x.to(self.device), y.to(self.device)
                        with torch.autocast(device_type=self.device, dtype=torch.float16):
                            _, loss = self.model(x, y) 
                        eval_loss += loss.item()
                
                b = time.time()
                new_row = pd.DataFrame(data={
                    "iter": [itr],
                    "train_loss": [running_loss/(self.eval_step*self.accum_steps)],
                    "eval_loss": [eval_loss/self.eval_iters],
                    "lr": [lr],
                    "time": [round(b-a, 3)]
                })
                running_loss = 0
                self.metrics = pd.concat([self.metrics, new_row], axis=0, ignore_index=True)
                print(self.metrics.to_string(index=False))
                a = time.time()
            
            self.model.train()
                    

    def save(self, file_path):
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model_config": self.model_config,
            "train_config": self.train_config
        }
        torch.save(checkpoint, file_path)

    def plot_metrics(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.metrics['iter'], self.metrics['train_loss'], label='Train Loss')
        plt.plot(self.metrics['iter'], self.metrics['test_loss'], label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Test Loss Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()