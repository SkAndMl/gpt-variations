import matplotlib.pyplot as plt
import torch
import time
import pandas as pd
from torch.optim import AdamW
from torch.utils.data import DataLoader
import numpy as np
from gpts.data import TextDataset
from gpts.models import GPT, pGPT, ccGPT, lcGPT

class Trainer:

    def __init__(self,
                 model_config,
                 model="gpt",
                 lr=3e-4,
                 bs=32,
                 epochs=10,
                 device="cpu"):
        
        self.model_config = model_config
        self.device = device
        self._init_model(model)
        self.optimizer = AdamW(params=self.model.parameters(),
                                lr=lr)
        self._init_data(bs)
        self.epochs = epochs
        self.metrics = pd.DataFrame(columns=["epochs", "train_loss", "test_loss", "epoch_time"])
        

    
    def _init_model(self, model):

        if model == "gpt":
            self.model = GPT(self.model_config)
        elif model == "pgpt":
            self.model = pGPT(self.model_config)
        elif model == "ccgpt":
            self.model = ccGPT(self.model_config)
        else:
            self.model = lcGPT(self.model_config)
        
        self.model = self.model.to(self.device)
        
    
    def _init_data(self, bs):
        train_ds = TextDataset(train=True)
        test_ds = TextDataset(train=False)
        self.train_dl = DataLoader(dataset=train_ds,
                                   batch_size=bs, 
                                   shuffle=True)
        self.test_dl = DataLoader(dataset=test_ds,
                                  batch_size=bs,
                                  shuffle=False)


    def _train(self):
        self.model.train()
        tot_loss = 0
        for x, y in self.train_dl:
            x, y = x.to(self.device), y.to(self.device)
            _, loss = self.model(x, y)
            tot_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        return tot_loss/len(self.train_dl)


    @torch.inference_mode()
    def _eval(self):
        self.model.eval()
        tot_loss = 0
        for x, y in self.test_dl:
            x, y = x.to(self.device), y.to(self.device)
            _, loss = self.model(x, y)
            tot_loss += loss.item()
        
        return tot_loss/len(self.test_dl)
 
    
    def fit(self):

        for epoch in range(1, self.epochs+1):

            a = time.time()
            train_loss, test_loss = self._train(), self._eval()
            b = time.time()

            new_row = pd.DataFrame(data = {
                "epochs": [epoch],
                "train_loss": [train_loss],
                "test_loss": [test_loss],
                "epoch_time": [round(b-a, 3)]
            })
            self.metrics = pd.concat([self.metrics, new_row], axis=0, ignore_index=True)

            print(self.metrics.to_string(index=False))
        
        return self.metrics
    
    def save(self, file_path):
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model_config": self.model_config
        }

        torch.save(checkpoint, file_path)

    def plot_metrics(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.metrics['epoch'], self.metrics['train_loss'], label='Train Loss')
        plt.plot(self.metrics['epoch'], self.metrics['test_loss'], label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Test Loss Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
