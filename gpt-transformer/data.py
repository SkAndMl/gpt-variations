from torch.utils.data import Dataset
from tokenizers import Tokenizer
import torch
import random
import os

class Tokens:
    pad_token = 0
    sos_token = 1
    eos_token = 2
    unk_token = 3
    sep_token = 4
    

def split_data(lang1: str, lang2: str, test_size: float=0.1) -> None:

    with open(f"{lang1}-{lang2}.txt", "r") as f:
        data = f.readlines()
        data = [d.strip() for d in data]
    
    random.seed(2406)
    random.shuffle(data)

    split_idx = int(len(data)*test_size)
    with open(f"{lang1}-{lang2}-test.txt", "w") as f:
        for d in data[:split_idx]:
            f.write(d+"\n")
    
    with open(f"{lang1}-{lang2}-train.txt", "w") as f:
        for d in data[split_idx:]:
            f.write(d+"\n")

class LangDataset(Dataset):

    def __init__(self, lang1: str, lang2: str, split: str="train", test_size: float=0.1) -> None:
        super().__init__()

        self.lang1 = lang1
        self.lang2 = lang2
        self.tokenizer: Tokenizer = Tokenizer.from_file(f"{lang1}-{lang2}.json")
        
        if not os.path.exists(f"{lang1}-{lang2}-{split}.txt"):
            split_data(lang1, lang2, test_size)
        
        with open(f"{lang1}-{lang2}-{split}.txt", "r") as f:
            data = f.readlines()
            data = [d.strip() for d in data]
        
        self.data = data
    

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> torch.Tensor:
        
        assert index < len(self.data), IndexError(f"{index} must be less than {len(self.data)}")
        
        tokens = torch.tensor(self.tokenizer.encode(self.data[index]).ids, dtype=torch.long)
        return tokens

    @staticmethod
    def calc_max_seq_len(lang1: str, lang2: str) -> int:
        with open(f"{lang1}-{lang2}.txt", "r") as f:
            data = f.readlines()
            data = [d.strip() for d in data]
        
        tokenizer: Tokenizer = Tokenizer.from_file(f"{lang1}-{lang2}.json")
        max_len = 0
        for i in range(len(data)):
            tokens = len(tokenizer.encode(data[i]).ids) 
            max_len = max(max_len, tokens)
        return max_len


if __name__ == "__main__":
    ds = LangDataset("en", "fr", "train")
    test_ds = LangDataset("en", "fr", "test")
    print("loaded both")
    print(LangDataset.calc_max_seq_len("en", "fr"))