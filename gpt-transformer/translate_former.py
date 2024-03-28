import torch
from torch import nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from typing import Optional, Tuple, Dict, Union
from scripts import Tokens, Embedding, Decoder
from scripts import ConvDecoder, PosDecoder, PosEmbedding


class TranslateFormer(nn.Module):

    def __init__(self, config: Dict[str, Union[int,str]]) -> None:

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
    def translate(self, x: str, max_len: int=20) -> str:
        x = "<sos>" + x + "<sep>"
        tokens = torch.tensor([self.tokenizer.encode(x).ids], dtype=torch.long, requires_grad=False,
                              device=self.device)

        while tokens[-1, -1].item() != Tokens.eos_token and tokens.size(-1)<max_len:

            logits, _ = self(tokens) # B, T+1, VOCAB_SIZE
            max_token_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True).to(self.device)

            tokens = torch.cat([tokens, max_token_id], dim=-1)
        
        op: str = self.tokenizer.decode(list(tokens.detach().cpu().numpy()[0]), skip_special_tokens=False)
        sep_idx = op.find("<sep>")
        predicted = op[sep_idx+5:]
        if "<eos>" in predicted:
            predicted = predicted[:-5]
        return predicted.strip()
    

class ParallelTranslateFormer(nn.Module):

    def __init__(self, config: Dict[str, Union[int,str]]) -> None:

        super().__init__()
        self.input_embedding = Embedding(config)
        config["n_layers"] //= 2
        self.decoder_1 = Decoder(config)
        self.decoder_2 = Decoder(config)
        self.weight = nn.Parameter(data=torch.tensor(0.5), requires_grad=True)
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
    def translate(self, x: str, max_len: int=20) -> str:
        x = "<sos>" + x + "<sep>"
        tokens = torch.tensor([self.tokenizer.encode(x).ids], dtype=torch.long, requires_grad=False,
                              device=self.device)

        while tokens[-1, -1].item() != Tokens.eos_token and tokens.size(-1)<max_len:

            logits, _ = self(tokens) # B, T+1, VOCAB_SIZE
            max_token_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True).to(self.device)

            tokens = torch.cat([tokens, max_token_id], dim=-1)
        
        op: str = self.tokenizer.decode(list(tokens.detach().cpu().numpy()[0]), skip_special_tokens=False)
        sep_idx = op.find("<sep>")
        predicted = op[sep_idx+5:]
        if "<eos>" in predicted:
            predicted = predicted[:-5]
        return predicted.strip()
    

class ConvTranslateFormer(nn.Module):

    def __init__(self, config: Dict[str, Union[str, int]]) -> None:

        super().__init__()
        d_model = config["d_model"]
        self.input_embedding = Embedding(config)
        self.conv_decoder = ConvDecoder(config=config)
        self.cls_net = nn.Sequential(
            nn.Dropout(config["dropout"]),
            nn.Linear(d_model//(2**(config["n_layers"]//2)),
                      config["vocab_size"])
        )
        self.tokenizer = Tokenizer.from_file(config["tokenizer_file_path"])
        self.device = config["device"]
    

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor]:

        x = self.input_embedding(x)
        decoder_out = self.conv_decoder(x)
        logits: torch.Tensor = self.cls_net(decoder_out) # B, K, OUTPUT_VOCAB_SIZE

        B, SEQ_LEN, _ = logits.shape
        loss = None
        if y is not None:
            loss = F.cross_entropy(logits.reshape(B*SEQ_LEN, -1), target=y.reshape(B*SEQ_LEN,),
                                   ignore_index=Tokens.pad_token)
        
        return logits, loss
    
    @torch.inference_mode()
    def translate(self, x: str, max_len: int=20) -> str:
        x = "<sos>" + x + "<sep>"
        tokens = torch.tensor([self.tokenizer.encode(x).ids], dtype=torch.long, requires_grad=False,
                              device=self.device)

        while tokens[-1, -1].item() != Tokens.eos_token and tokens.size(-1)<max_len:

            logits, _ = self(tokens) # B, T+1, VOCAB_SIZE
            max_token_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True).to(self.device)

            tokens = torch.cat([tokens, max_token_id], dim=-1)
        
        op: str = self.tokenizer.decode(list(tokens.detach().cpu().numpy()[0]), skip_special_tokens=False)
        sep_idx = op.find("<sep>")
        predicted = op[sep_idx+5:]
        if "<eos>" in predicted:
            predicted = predicted[:-5]
        return predicted.strip()
    

class PosTranslateFormer(nn.Module):

    def __init__(self, config: Dict[str, Union[int,str]]) -> None:

        super().__init__()
        self.input_embedding = PosEmbedding(config)
    
        self.decoder = PosDecoder(config)
        self.cls_net = nn.Sequential(
            nn.Dropout(config["dropout"]),
            nn.Linear(in_features=config["d_model"], out_features=config["vocab_size"])
        )
        self.tokenizer = Tokenizer.from_file(config["tokenizer_file_path"])
        self.device = config["device"]

    
    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor]:

        x, pos_embedding = self.input_embedding(x)
        decoder_out = self.decoder(x, pos_embedding)
        logits: torch.Tensor = self.cls_net(decoder_out) # B, K, OUTPUT_VOCAB_SIZE

        B, SEQ_LEN, _ = logits.shape
        loss = None
        if y is not None:
            loss = F.cross_entropy(logits.reshape(B*SEQ_LEN, -1), target=y.reshape(B*SEQ_LEN,),
                                   ignore_index=Tokens.pad_token)
        
        return logits, loss
    
    @torch.inference_mode()
    def translate(self, x: str, max_len: int=20) -> str:
        x = "<sos>" + x + "<sep>"
        tokens = torch.tensor([self.tokenizer.encode(x).ids], dtype=torch.long, requires_grad=False,
                              device=self.device)

        while tokens[-1, -1].item() != Tokens.eos_token and tokens.size(-1)<max_len:

            logits, _ = self(tokens) # B, T+1, VOCAB_SIZE
            max_token_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True).to(self.device)

            tokens = torch.cat([tokens, max_token_id], dim=-1)
        
        op: str = self.tokenizer.decode(list(tokens.detach().cpu().numpy()[0]), skip_special_tokens=False)
        sep_idx = op.find("<sep>")
        predicted = op[sep_idx+5:]
        if "<eos>" in predicted:
            predicted = predicted[:-5]
        return predicted.strip()