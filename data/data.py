from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from datasets import load_dataset
import torch
from typing import Tuple, Union
from torch.nn.utils.rnn import pad_sequence

batch_size = 32

class Tokens:
    pad_token = 0
    sos_token = 1
    eos_token = 2
    unk_token = 3

class LangDataset(Dataset):

    def __init__(self, lang1: str, lang2: str, split: str="train") -> None:
        super().__init__()

        self.lang1 = lang1
        self.lang2 = lang2
        self.tokenizer = Tokenizer.from_file(f"tokenizer/{lang1}-{lang2}.json")
        self.dataset = load_dataset("opus_books", f"{lang1}-{lang2}")["train"].train_test_split(test_size=0.1, shuffle=True, seed=2406)[split]
    
        self._calc_max_seq_len()

    def __len__(self):
        return self.dataset.num_rows

    def __getitem__(self, index) -> torch.Tensor:
        
        assert index < self.dataset.num_rows, IndexError(f"{index} must be less than {self.dataset.num_rows}")
        
        lang1_sentence = self.dataset[index]["translation"][self.lang1]
        lang2_sentence = self.dataset[index]["translation"][self.lang2]
        lang1_sentence = "<sos> " + lang1_sentence + " <eos>"
        lang2_sentence = "<sos> " + lang2_sentence + " <eos>"
        tokens = torch.tensor(self.tokenizer.encode(lang1_sentence+lang2_sentence).ids, dtype=torch.long)

        return tokens

    def _calc_max_seq_len(self):
        self.max_seq_len = 0
        for i in range(self.dataset.num_rows):
            tokens = self[i]
            self.max_seq_len = max(self.max_seq_len, tokens.shape[0])


def collate_fn(batch) -> Tuple[torch.Tensor]:

    en, fr = [], []
    for (en_, fr_) in batch:
        en.append(en_)
        fr.append(fr_)
    

    en = pad_sequence(sequences=en, batch_first=True, padding_value=Tokens.pad_token)
    fr = pad_sequence(sequences=fr, batch_first=True, padding_value=Tokens.pad_token)
    return en, fr

def create_dataloader(lang1: str, lang2: str, split: str="train") -> Tuple[Union[DataLoader, int]]:

    dataset = LangDataset(lang1=lang1, lang2=lang2, split=split)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True if split=="train" else False,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    return dataloader, dataset.max_seq_len

if __name__ == "__main__":
    dl, max_seq_len = create_dataloader("en", "fr", "train")
    print(max_seq_len)