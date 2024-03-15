from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import os
from datasets import load_dataset, Dataset
from model import Tokens

def create_file(ds: Dataset, lang: str) -> None:

    FILE_PATH = f"{lang}.txt"
    if not os.path.exists(FILE_PATH):
        with open(FILE_PATH, "w", encoding="utf-8") as f:
            for i in range(ds.num_rows):
                f.write(Tokens.sos + ds[i]["translation"][lang]+Tokens.eos+"\n")

def create_tokenizer(lang: str) -> None:
    
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()

    special_tokens = ["<pad>", "<sos>", "<eos>", "<unk>"]  
    trainer = BpeTrainer(special_tokens=special_tokens, vocab_size=30000, min_frequency=2) 

    tokenizer.train(files=[f"{lang}.txt"], trainer=trainer)
    tokenizer.save(path=f"{lang}.json")
    print(f"{lang}'s vocab size is {tokenizer.get_vocab_size()}")

def prepare_tokenizer(lang1: str, lang2: str) -> None:

    books = load_dataset("opus_books", f"{lang1}-{lang2}")
    books_ds = books["train"].train_test_split(test_size=0.1, shuffle=True, seed=2406)["train"]

    create_file(books_ds, lang1)
    create_file(books_ds, lang2)

    create_tokenizer(lang1)
    create_tokenizer(lang2)
   

if __name__ == "__main__":
    prepare_tokenizer(lang1="en", lang2="it")