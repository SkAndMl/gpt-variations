import torch
from torch import nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from typing import Optional, Tuple
from scripts import Tokens, Embedding, Decoder


class TranslateFormer(nn.Module):

    def __init__(self, config) -> None:

        super().__init__()
        self.input_embedding = Embedding(config)
    
        self.decoder = Decoder(config)
        self.cls_net = nn.Sequential(
            nn.Dropout(config["dropout"]),
            nn.Linear(in_features=config["d_model"], out_features=config["vocab_size"])
        )
        self.tokenizer = Tokenizer.from_file(config["tokenizer_file_path"])
        self.device = config["device"]

    
    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor]:

        x = self.input_embedding(x)
        decoder_out = self.decoder(x)
        logits: torch.Tensor = self.cls_net(decoder_out) # B, K, OUTPUT_VOCAB_SIZE

        B, SEQ_LEN, _ = logits.shape
        loss = None
        if y is not None:
            loss = F.cross_entropy(logits.reshape(B*SEQ_LEN, -1), target=y.reshape(B*SEQ_LEN,),
                                   ignore_index=Tokens.pad_token)
        
        return logits, loss
    
    @torch.inference_mode()
    def translate(self, x: str, max_new_tokens: int=20) -> str:
        x = "<sos>" + x + "<sep>"

        num_new_tokens = 0
        tokens = torch.tensor(self.tokenizer.encode(x).ids, dtype=torch.long, device=self.device).unsqueeze(0)
        while True:
            logits, _ = self(tokens)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_token_id = torch.argmax(probs, dim=-1)
            tokens = torch.cat([tokens, next_token_id.unsqueeze(0)], dim=-1)
            num_new_tokens += 1
            if next_token_id==Tokens.eos_token or num_new_tokens>max_new_tokens:
                break
        

        return self.tokenizer.decode(list(tokens.numpy()[0]))
    

class ParallelTranslateFormer(nn.Module):

    def __init__(self, config) -> None:

        super().__init__()
        self.input_embedding = Embedding(config)
        config["n_layers"] //= 2
        self.decoder_1 = Decoder(config)
        self.decoder_2 = Decoder(config)
        self.weight = nn.Parameter(data=0.5, requires_grad=True)
        self.cls_net = nn.Sequential(
            nn.Dropout(config["dropout"]),
            nn.Linear(in_features=config["d_model"], out_features=config["vocab_size"])
        )
        self.tokenizer = Tokenizer.from_file(config["tokenizer_file_path"])
        self.device = config["device"]

    
    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor]:

        x = self.input_embedding(x)
        decoder_1_out = self.decoder_1(x)
        decoder_2_out = self.decoder_2(x)
        logits: torch.Tensor = self.cls_net(self.weight*decoder_1_out + (1-self.weight)*decoder_2_out) # B, K, OUTPUT_VOCAB_SIZE

        B, SEQ_LEN, _ = logits.shape
        loss = None
        if y is not None:
            loss = F.cross_entropy(logits.reshape(B*SEQ_LEN, -1), target=y.reshape(B*SEQ_LEN,),
                                   ignore_index=Tokens.pad_token)
        
        return logits, loss
    
    @torch.inference_mode()
    def translate(self, x: str, max_new_tokens: int=20) -> str:
        x = "<sos>" + x + "<sep>"

        num_new_tokens = 0
        tokens = torch.tensor(self.tokenizer.encode(x).ids, dtype=torch.long, device=self.device).unsqueeze(0)
        while True:
            logits, _ = self(tokens)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_token_id = torch.argmax(probs, dim=-1)
            tokens = torch.cat([tokens, next_token_id.unsqueeze(0)], dim=-1)
            num_new_tokens += 1
            if next_token_id==Tokens.eos_token or num_new_tokens>max_new_tokens:
                break
        

        return self.tokenizer.decode(list(tokens.numpy()[0]))    

