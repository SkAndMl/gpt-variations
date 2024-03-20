from torch.utils.data import Dataset
from tokenizers import Tokenizer
from datasets import load_dataset
import torch

class Tokens:
    pad_token = 0
    sos_token = 1
    eos_token = 2
    unk_token = 3
    sep_token = 4

class LangDataset(Dataset):

    def __init__(self, lang1: str, lang2: str, split: str="train") -> None:
        super().__init__()

        self.lang1 = lang1
        self.lang2 = lang2
        self.tokenizer = Tokenizer.from_file(f"{lang1}-{lang2}.json")
        self.dataset = load_dataset("opus_books", f"{lang1}-{lang2}")["train"].train_test_split(test_size=0.1, shuffle=True, seed=2406)[split]
    

    def __len__(self):
        return self.dataset.num_rows

    def __getitem__(self, index) -> torch.Tensor:
        
        assert index < self.dataset.num_rows, IndexError(f"{index} must be less than {self.dataset.num_rows}")
        
        lang1_sentence = self.dataset[index]["translation"][self.lang1]
        lang2_sentence = self.dataset[index]["translation"][self.lang2]
        lang1_sentence = "<sos> " + lang1_sentence + " <sep>"
        lang2_sentence = lang2_sentence + " <eos>"
        tokens = torch.tensor(self.tokenizer.encode(lang1_sentence+lang2_sentence).ids, dtype=torch.long)

        return tokens

    @staticmethod
    def calc_max_seq_len(lang1: str, lang2: str) -> int:
        dataset = load_dataset("opus_books", f"{lang1}-{lang2}")["train"]
        tokenizer: Tokenizer = Tokenizer.from_file(f"{lang1}-{lang2}.json")
        max_len = 0
        for i in range(dataset.num_rows):
            tokens = len(tokenizer.encode(dataset[i]["translation"][lang1]).ids) + \
                     len(tokenizer.encode(dataset[i]["translation"][lang2]).ids) + 3
            max_len = max(max_len, tokens)
        return max_len


if __name__ == "__main__":
    ds = LangDataset("en", "it", "train")
    test_ds = LangDataset("en", "it", "test")
    print("loaded both")
    print(LangDataset.calc_max_seq_len("en", "it"))