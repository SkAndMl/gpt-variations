from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from datasets import load_dataset
import torch
from typing import Tuple, Union, Dict
from torch.nn.utils.rnn import pad_sequence

batch_size = 32

class Tokens:
    pad_token = 0
    sos_token = 1
    eos_token = 2
    unk_token = 3

class LangDataset(Dataset):

    def __init__(self, split: str="train") -> None:
        super().__init__()

        self.en_tokenizer = Tokenizer.from_file("tokenizer/en.json")
        self.fr_tokenizer = Tokenizer.from_file("tokenizer/fr.json")
        self.dataset = load_dataset("opus_books", "en-fr")["train"].train_test_split(test_size=0.1, shuffle=True, seed=2406)[split]
    
        self._calc_max_seq_len()

    def __len__(self):
        return self.dataset.num_rows

    def __getitem__(self, index) -> Tuple[torch.Tensor]:
        
        assert index < self.dataset.num_rows, IndexError(f"{index} must be less than {self.dataset.num_rows}")

        en_sentence = self.dataset[index]["translation"]["en"]
        fr_sentence = self.dataset[index]["translation"]["fr"]
        en_sentence = "<sos> " + en_sentence + " <eos>"
        fr_sentence = "<sos> " + fr_sentence + " <eos>"
        en_tokens = torch.tensor(self.en_tokenizer.encode(en_sentence).ids, dtype=torch.long)
        fr_tokens = torch.tensor(self.fr_tokenizer.encode(fr_sentence).ids, dtype=torch.long)

        return en_tokens, fr_tokens

    def _calc_max_seq_len(self):
        self.lang_max_seq_len = {
            "en" : 0,
            "fr" : 0
        }
        for i in range(self.dataset.num_rows):
            en_tokens, fr_tokens = self[i]
            self.lang_max_seq_len["en"] = max(self.lang_max_seq_len["en"], en_tokens.shape[0])
            self.lang_max_seq_len["fr"] = max(self.lang_max_seq_len["fr"], fr_tokens.shape[0])


def collate_fn(batch) -> Tuple[torch.Tensor]:

    en, fr = [], []
    for (en_, fr_) in batch:
        en.append(en_)
        fr.append(fr_)
    

    en = pad_sequence(sequences=en, batch_first=True, padding_value=Tokens.pad_token)
    fr = pad_sequence(sequences=fr, batch_first=True, padding_value=Tokens.pad_token)
    return en, fr

def create_dataloader(split: str="train") -> Tuple[Union[DataLoader, Dict[str, int]]]:

    dataset = LangDataset(split=split)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True if split=="train" else False,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    return dataloader, dataset.lang_max_seq_len

if __name__ == "__main__":
    ds = LangDataset(split="train")
    print(ds[0])