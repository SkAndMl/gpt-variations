import torch
from torch.utils.data import Dataset
import numpy as np

class TextDataset(Dataset):

    def __init__(self, train=True):

        file_path = "data/test.npy"
        if train:
            file_path = "data/train.npy"
        
        self.data = np.load(file_path)

    def __len__(self): return self.data.shape[0]

    def __getitem__(self, idx):
        assert idx<self.data.shape[0], IndexError("idx bigger than size of ds")
        return torch.tensor(self.data[idx, :-1]), torch.tensor(self.data[idx, 1:])