import torch
from torch import nn
import torch.nn.functional as F
import math
from tokenizers import Tokenizer
from typing import Optional, Tuple

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

        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")

        if not self.flash:
            mask = torch.ones(size=(1, 1, config["seq_len"], config["seq_len"])).tril(diagonal=0)
            self.register_buffer(name="mask", tensor=mask)


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, T, D_MODEL = x.shape
        q, k, v = self.proj(x).split(D_MODEL, dim=2) # B, T, D_MODEL
        q = q.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2) 
        k = k.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if self.flash:
            attn_outputs: torch.Tensor = F.scaled_dot_product_attention(query=q, key=k, value=v,
                                                          is_causal=True, dropout_p=self.dropout_p)
        else:
            attn_outputs: torch.Tensor = q @ k.transpose(-2, -1)
            attn_outputs = attn_outputs.masked_fill(self.mask[:, :, :T, :T].logical_not(), value=float("-inf"))
            attn_outputs = F.softmax(attn_outputs/math.sqrt(self.head_dim))
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