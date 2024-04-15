import torch
from torch import nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from typing import Optional, Tuple, Dict, Union
from scripts import Embedding, Decoder
from scripts import ConvDecoder


class VanillaGPT(nn.Module):

    def __init__(self, config: Dict[str, Union[int,str]]) -> None:

        super().__init__()
        self.input_embedding = Embedding(config)
        self.vocab_size = config["vocab_size"]
        self.context_length = config["context_length"]
        self.decoder = Decoder(config)
        self.cls_net = nn.Sequential(
            nn.Dropout(config["dropout"]),
            nn.Linear(in_features=config["d_model"], out_features=config["vocab_size"])
        )
        self.device = config["device"]

    
    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor]:

        x = self.input_embedding(x)
        decoder_out = self.decoder(x)
        logits: torch.Tensor = self.cls_net(decoder_out) # B, K, OUTPUT_VOCAB_SIZE

        B, SEQ_LEN, _ = logits.shape
        loss = None
        if y is not None:
            loss = F.cross_entropy(logits.reshape(B*SEQ_LEN, -1), target=y.reshape(B*SEQ_LEN,))
        
        return logits, loss
    
    @torch.inference_mode()
    def translate(self, x: torch.Tensor=None, max_len: int=20) -> torch.Tensor:

        if x is None:
            x = torch.randint(low=0, high=self.vocab_size-1).resize((1, 1))
        
        x = x.to(self.device)
        
        while x.shape[-1] < max_len:
            logits, _ = self(x[:, -self.context_length:])
            max_token_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True).to(self.device)
            x = torch.cat([x, max_token_id], dim=-1)

        return x
        
    

class ParallelGPT(nn.Module):

    def __init__(self, config: Dict[str, Union[int,str]]) -> None:

        super().__init__()
        config["d_model"] *= 2
        self.input_embedding = Embedding(config)
        config["d_model"] //= 2
        config["n_layers"] //= 2
        
        self.decoder_1 = Decoder(config)
        self.decoder_2 = Decoder(config)
        self.weight = nn.Parameter(data=torch.tensor(0.5), requires_grad=True)
        self.cls_net = nn.Sequential(
            nn.Dropout(config["dropout"]),
            nn.Linear(in_features=config["d_model"], out_features=config["vocab_size"])
        )

        self.d_model = config["d_model"]
        self.vocab_size = config["vocab_size"]
        self.context_length = config["context_length"]
        self.device = config["device"]

    
    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor]:

        x = self.input_embedding(x)
        x1, x2 = x.split(split_size=self.d_model, dim=-1)
        decoder_1_out = self.decoder_1(x1)
        decoder_2_out = self.decoder_2(x2)
        logits: torch.Tensor = self.cls_net(self.weight*decoder_1_out + (1-self.weight)*decoder_2_out) # B, K, OUTPUT_VOCAB_SIZE

        B, SEQ_LEN, _ = logits.shape
        loss = None
        if y is not None:
            loss = F.cross_entropy(logits.reshape(B*SEQ_LEN, -1), target=y.reshape(B*SEQ_LEN,))
        
        return logits, loss
    
## TODO: add generate function to parallelformer
    

class ConvGPT(nn.Module):

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

        self.vocab_size = config["vocab_size"]
        self.context_length = config["context_length"]        
        self.device = config["device"]
    

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor]:

        x = self.input_embedding(x)
        decoder_out = self.conv_decoder(x)
        logits: torch.Tensor = self.cls_net(decoder_out) # B, K, OUTPUT_VOCAB_SIZE

        B, SEQ_LEN, _ = logits.shape
        loss = None
        if y is not None:
            loss = F.cross_entropy(logits.reshape(B*SEQ_LEN, -1), target=y.reshape(B*SEQ_LEN,))
        
        return logits, loss
    
    @torch.inference_mode()
    def translate(self, x: torch.Tensor=None, max_len: int=20) -> torch.Tensor:

        if x is None:
            x = torch.randint(low=0, high=self.vocab_size-1).resize((1, 1))
        
        x = x.to(self.device)
        
        while x.shape[-1] < max_len:
            logits, _ = self(x[:, -self.context_length:])
            max_token_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True).to(self.device)
            x = torch.cat([x, max_token_id], dim=-1)

        return x