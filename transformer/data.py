import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from tokenizers import Tokenizer
from typing import Dict, List
from model import Tokens

class LangDataset(Dataset):

    def __init__(self, lang1: str, lang2: str, split: bool="train") -> None:
        super().__init__()

        self.lang1 = lang1
        self.lang2 = lang2
        self.ds: DatasetDict = load_dataset("opus_books", f"{lang1}-{lang2}")
        self.ds = self.ds["train"].train_test_split(test_size=0.1, shuffle=True, seed=2406)[split]
        self.lang1_tokenizer: Tokenizer = Tokenizer.from_file(f"{lang1}.json")
        self.lang2_tokenizer: Tokenizer = Tokenizer.from_file(f"{lang2}.json")

        self._calc_max_seq_len()

    def __len__(self) -> int:
        return self.ds.num_rows

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        
        lang1_tokens: List[int] = self.lang1_tokenizer.encode(self.ds[index]["translation"][self.lang1]).ids
        lang2_tokens: List[int] = self.lang2_tokenizer.encode(self.ds[index]["translation"][self.lang2]).ids

        lang1_tokens = [Tokens.sos_token] + lang1_tokens + [Tokens.eos_token]
        lang1_num_pad_tokens = self.max_seq_len - len(lang1_tokens)
        lang1_tokens = torch.tensor(lang1_tokens+[Tokens.pad_token]*lang1_num_pad_tokens, dtype=torch.long)
        
        label = lang2_tokens + [Tokens.eos_token]
        lang2_tokens = [Tokens.sos_token] + lang2_tokens
        lang2_num_pad_tokens = self.max_seq_len - len(lang2_tokens)
        lang2_tokens = torch.tensor(lang2_tokens+[Tokens.pad_token]*lang2_num_pad_tokens, dtype=torch.long)
        label = torch.tensor(label+[Tokens.pad_token]*lang2_num_pad_tokens, dtype=torch.long)

        return {
            "encoder_input" : lang1_tokens,
            "decoder_input" : lang2_tokens,
            "label" : label,
            "encoder_mask" : (lang1_tokens!=Tokens.pad_token).int().unsqueeze(0).unsqueeze(0),
            "decoder_mask" : ((lang2_tokens!=Tokens.pad_token).int() & self._create_causal_mask(lang2_tokens.shape[-1])).int().unsqueeze(0)
        }

    def _calc_max_seq_len(self) -> None:

        self.max_seq_len = 0
        for i in range(self.ds.num_rows):
            lang1_tokens = self.lang1_tokenizer.encode(self.ds[i]["translation"][self.lang1]).ids
            lang2_tokens = self.lang2_tokenizer.encode(self.ds[i]["translation"][self.lang2]).ids

            self.max_seq_len = max(self.max_seq_len, len(lang1_tokens), len(lang2_tokens))
    

    def _create_causal_mask(self, size: int) -> torch.Tensor:
        return torch.ones(size=(size, size), dtype=torch.long).tril(diagonal=0)
    

if __name__ == "__main__":

    ds = LangDataset("en", "it")
    op = ds[0]
    print(f"Encoder input: {op['encoder_input'].shape}")
    print(f"Decoder input: {op['decoder_input'].shape}")
    print(f"Label: {op['label'].shape}")
    print(f"Encoder mask: {op['encoder_mask'].shape}")
    print(f"Decoder mask: {op['decoder_mask'].shape}")