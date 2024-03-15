from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import os
from datasets import load_dataset

def prepare_tokenizer(lang1: str, lang2: str) -> None:

    books = load_dataset("opus_books", f"{lang1}-{lang2}")
    books_ds = books["train"].train_test_split(test_size=0.1, shuffle=True, seed=2406)

    FILE_PATH = f"data/{lang1}-{lang2}.txt"
    if not os.path.exists(FILE_PATH):
        with open(FILE_PATH, 'w', encoding='utf-8') as f:
            for i in range(books_ds["train"].num_rows):
                lang1_sentence = "<sos>" + books_ds["train"][i]["translation"][lang1] + "<sep>"
                lang2_sentence = books_ds["train"][i]["translation"][lang2] + "<eos>" + "\n"
                f.write(lang1_sentence + lang2_sentence)

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()

    special_tokens = ["<pad>", "<sos>", "<eos>", "<unk>", "<sep>"]  
    trainer = BpeTrainer(special_tokens=special_tokens, vocab_size=30000, min_frequency=True) 

    tokenizer.train(files=[f"data/{lang1}-{lang2}.txt"], trainer=trainer)

    if not os.path.exists("tokenizer"):
        os.makedirs("tokenizer")

    tokenizer.save(path=f"tokenizer/{lang1}-{lang2}.json")

    print(f"{lang1}-{lang2}'s vocab size is: {tokenizer.get_vocab_size()}")

if __name__ == "__main__":
    prepare_tokenizer(lang1="en", lang2="it")