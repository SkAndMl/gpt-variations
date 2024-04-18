from datasets import load_dataset
from itertools import islice

split_sizes = {
    "train": 10000,
    "valid": 1000
}

for split in split_sizes:
    dataset = load_dataset(f"transformersbook/codeparrot-{split}", split="validation" if split=="valid" else split, streaming=True)
    split_rows = list(islice(dataset, split_sizes[split]))

    with open(f"{split}.txt", "w") as f:
        for row in split_rows:
            f.write(row["content"]+"\n")