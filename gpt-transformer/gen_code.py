import torch
import json
import time
import argparse
from model import GPT
device = "cuda" if torch.cuda.is_available() else "cpu"

with open("gpt-transformer/config.json") as f:
    config = json.load(f)

with open("gpt-transformer/train.txt", "r", encoding="utf-8") as train, open("gpt-transformer/valid.txt", "r", encoding="utf-8") as valid:
    data = train.read() + valid.read()
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    config["vocab_size"] = vocab_size
    config["device"] = device
    itos = {i: ch for i, ch in enumerate(chars)}
    stoi = {ch: i for i, ch in enumerate(chars)}

decode = lambda l: "".join([itos[i] for i in l]) 
encode = lambda s: [stoi[ch] for ch in s]

gpt = GPT(config=config)
gpt.load_state_dict(torch.load("checkpoints/vanillagpt.pt",
                                map_location=torch.device(device=device)))


def print_like_gpt(text):

    for ch in text:
        print(ch, end="", flush=True,)
        time.sleep(0.02)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_tokens", required=True, metavar="", type=int)
    parser.add_argument("-s", "--string", required=False, metavar="", type=str, default=None)
    args = parser.parse_args()
    input_string = args.string
    x = None
    if input_string is not None:
        x = torch.tensor(encode(input_string), dtype=torch.long).view(1, -1)
    out = gpt.generate(max_new_tokens=args.num_tokens, x=x) # [1, S]
    text = decode(out[0].cpu().numpy())
    print_like_gpt(text=text)