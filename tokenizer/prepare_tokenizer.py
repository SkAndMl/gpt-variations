from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import os
from datasets import load_dataset

books = load_dataset("opus_books", "en-fr")
books_ds = books["train"].train_test_split(test_size=0.1, shuffle=True, seed=2406)


def prepare_tokenizer(lang: str) -> None:

    FILE_PATH = f"data/{lang}.txt"
    if not os.path.exists(FILE_PATH):
        with open(FILE_PATH, 'w', encoding='utf-8') as f:
            for i in range(books_ds["train"].num_rows):
                f.write(books_ds["train"][i]["translation"][lang] + '\n')

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()

    special_tokens = ["<pad>", "<sos>", "<eos>", "<unk>"]  
    trainer = BpeTrainer(special_tokens=special_tokens) 

    tokenizer.train(files=[f"data/{lang}.txt"], trainer=trainer)

    if not os.path.exists("tokenizer"):
        os.makedirs("tokenizer")

    tokenizer.save(path=f"tokenizer/{lang}.json")

    print(f"{lang}'s vocab size is: {tokenizer.get_vocab_size()}")

if __name__ == "__main__":
    prepare_tokenizer("en")
    prepare_tokenizer("fr")