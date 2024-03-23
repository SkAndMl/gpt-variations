from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import os
from datasets import load_dataset
import unicodedata
import re

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()

def prepare_tokenizer(lang1: str, lang2: str) -> None:

    books = load_dataset("opus_books", f"{lang1}-{lang2}")
    books_ds = books["train"].train_test_split(test_size=0.1, shuffle=True, seed=2406)

    count = 0
    FILE_PATH = f"{lang1}-{lang2}.txt"
    if not os.path.exists(FILE_PATH):
        with open(FILE_PATH, 'w', encoding='utf-8') as f:
            for i in range(books_ds["train"].num_rows):
                lang1_sentence = "<sos>" + normalize_string(books_ds["train"][i]["translation"][lang1]) + "<sep>"
                lang2_sentence = normalize_string(books_ds["train"][i]["translation"][lang2]) + "<eos>" + "\n"
                if len((lang1_sentence+lang1_sentence).split()) <= 20:
                    f.write(lang1_sentence + lang2_sentence)
                    count += 1
    print(f"Number of sentences: {count}")

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()

    special_tokens = ["<pad>", "<sos>", "<eos>", "<unk>", "<sep>"]  
    trainer = BpeTrainer(special_tokens=special_tokens, min_frequency=2) 

    tokenizer.train(files=[f"{lang1}-{lang2}.txt"], trainer=trainer)
    tokenizer.save(path=f"{lang1}-{lang2}.json")

    print(f"{lang1}-{lang2}'s vocab size is: {tokenizer.get_vocab_size()}")

if __name__ == "__main__":
    prepare_tokenizer(lang1="en", lang2="fr")