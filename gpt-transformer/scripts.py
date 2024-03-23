import torch
from torch import nn
import torch.nn.functional as F
import math
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Tokens:
    pad_token = 0
    sos_token = 1
    eos_token = 2
    unk_token = 3
    sep_token = 4


class Embedding(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.vocab_size = config["vocab_size"]
        self.d_model = config["d_model"]
        self.seq_len = config["seq_len"]
        self.device = config["device"]

        self.token_embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.d_model)
        self.pos_embedding = nn.Embedding(num_embeddings=self.seq_len, embedding_dim=self.d_model)
        self.dropout = nn.Dropout(p=config["dropout"])
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:

        B, T = x.shape

        assert T<=self.seq_len, AssertionError(f"Sequence length {T} should be less than or equal to {self.seq_len}")

        position = torch.arange(start=0, end=T, dtype=torch.int).unsqueeze(dim=0).to(self.device) # 1, T
        tok_emb = self.token_embedding(x) # B, T, D_MODEL
        pos_emb = self.pos_embedding(position) # 1, T, D_MODEL

        return self.dropout(tok_emb + pos_emb)
    

class MHA(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()

        assert config["d_model"]%config["n_heads"]==0, AssertionError(f"d_model: {config['d_model']} should be divisible by n_heads: {config['n_heads']}")

        self.n_heads = config["n_heads"]
        self.d_model = config["d_model"]
        self.head_dim = self.d_model//self.n_heads
        self.dropout_p = config["dropout"]
        
        self.proj = nn.Linear(in_features=self.d_model, out_features=self.d_model*3)
        self.o_proj = nn.Linear(in_features=self.d_model, out_features=self.d_model)
        self.dropout = nn.Dropout(p=self.dropout_p)


        mask = torch.ones(size=(1, 1, config["seq_len"], config["seq_len"]), dtype=torch.bool).tril(diagonal=0)
        self.register_buffer(name="mask", tensor=mask)


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, T, D_MODEL = x.shape
        q, k, v = self.proj(x).split(D_MODEL, dim=2) # B, T, D_MODEL
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2) 
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        attn_outputs: torch.Tensor = (q @ k.transpose(-2, -1))*(1/math.sqrt(k.size(-1)))
        attn_outputs = attn_outputs.masked_fill(self.mask[:, :, :T, :T]==0, value=float("-inf"))
        attn_outputs = F.softmax(attn_outputs, dim=-1)
        attn_outputs = self.dropout(attn_outputs) @ v
        
        attn_outputs = attn_outputs.transpose(1, 2).contiguous().view(B, T, D_MODEL)
        return self.o_proj(attn_outputs)
    

class FFN(nn.Module):

    def __init__(self, config):

        super().__init__()

        d_model = config["d_model"]

        self.net = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.GELU(),
            nn.Dropout(p=config["dropout"]),
            nn.Linear(d_model*4, d_model)
        )
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.net(x)
    

class DecoderBlock(nn.Module):

    def __init__(self, config) -> None:

        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=config["d_model"])
        self.masked_mha = MHA(config)
        self.ffn = FFN(config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        masked_out = self.layer_norm(x + self.masked_mha(x))
        ffn_out = self.layer_norm(masked_out + self.ffn(masked_out))
        return ffn_out

class Decoder(nn.Module):

    def __init__(self, config) -> None:

        super().__init__()

        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config["n_layers"])])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        for block in self.blocks:
            x = block(x)
        
        return x

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        nn.init.uniform_(m.weight, -1/math.sqrt(m.embedding_dim), 1/math.sqrt(m.embedding_dim))
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)