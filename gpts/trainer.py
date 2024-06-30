"""
script for training, checkpointing, inferencing, etc
June 30, 2024
"""

import matplotlib.pyplot as plt
import torch
import time
import pandas as pd
from torch.optim import AdamW
from gpts.data import TextDataset
from gpts.models import GPT, pGPT, ccGPT, lcGPT

class Trainer:

    def __init__(self, model_config, train_config):
        
        self.train_config = train_config
        self.model_config = model_config
        self.device = train_config["device"]
        self._init_model(model_config['model_type'])
        self.optimizer = AdamW(params=self.model.parameters(),
                                lr=train_config['lr'])
        self.train_ds = TextDataset(train=True)
        self.test_ds = TextDataset(train=False)
        self.train_iters = train_config['train_iters']
        self.eval_step = train_config['eval_step']
        self.eval_iters = train_config['eval_iters']
        self.accum_steps = train_config['gradient_accumulation_steps']
        self.metrics = pd.DataFrame(columns=["iter", "train_loss", "eval_loss", "time"])
        

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
        print(f"Model params: {params/10e6:.3f}")
        
        self.model = self.model.to(self.device)

 
    def _get_batch(self, train=True):
        
        if train:
            idxs = torch.randint(0, len(self.train_ds), size=(self.train_config['bs'],)).numpy()
            return self.train_ds[idxs]
        else:
            idxs = torch.randint(0, len(self.test_ds), size=(self.train_config['bs'],)).numpy()
            return self.test_ds[idxs]


    def fit(self):

        running_loss = 0
        a = time.time()
        for itr in range(1, self.train_iters+1):

            self.optimizer.zero_grad()
            
            x, y = self._get_batch(True)
            x, y = x.to(self.device), y.to(self.device)
            with torch.autocast(device_type=self.device, dtype=torch.float16):
                _, loss = self.model(x, y)

            running_loss += loss.item()
            loss /= self.accum_steps
            loss.backward()
            
            if itr%self.accum_steps==0:
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
                    "train_loss": [running_loss/self.eval_step],
                    "eval_loss": [eval_loss/self.eval_iters],
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