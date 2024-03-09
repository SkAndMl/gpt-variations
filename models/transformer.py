import torch
from torch import nn
import torch.nn.functional as F
import math
from typing import Tuple 

class Tokens:
    pad_token = 0

class Embedding(nn.Module):

    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int, dropout: float=0.1):

        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.pos_embedding = nn.Embedding(num_embeddings=max_seq_len, embedding_dim=d_model)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:

        B, T = x.shape

        assert T<=self.max_seq_len, AssertionError(f"Sequence length {T} should be less than or equal to {self.max_seq_len}")

        position = torch.arange(start=0, end=T, dtype=torch.int).unsqueeze(dim=0) # 1, T
        tok_emb = self.token_embedding(x) # B, T, D_MODEL
        pos_emb = self.pos_embedding(position) # 1, T, D_MODEL

        return self.dropout(tok_emb + pos_emb)


class MHA(nn.Module):

    def __init__(self, n_heads: int, d_model: int, dropout_p: float=0.1, is_causal: bool=False):
        
        super().__init__()

        assert d_model%n_heads == 0, AssertionError(f"{d_model} should be divisible by {n_heads}")

        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = d_model//n_heads
        self.is_causal = is_causal
        self.dropout_p = dropout_p


        self.q_proj = nn.Linear(in_features=d_model, out_features=self.head_dim)
        self.k_proj = nn.Linear(in_features=d_model, out_features=self.head_dim)
        self.v_proj = nn.Linear(in_features=d_model, out_features=self.head_dim)
        self.o_proj = nn.Linear(in_features=d_model, out_features=d_model)

        self.dropout = nn.Dropout(p=dropout_p)

        self.flash = False
        if hasattr(F.scaled_dot_product_attention):
            self.flash = True
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:

        # query -> B, Q, D_MODEL
        # key, value -> B, K, D_MODEL
        B, Q, D_MODEL = query.shape
        _, K, _ = key.shape

        q: torch.Tensor = self.q_proj(query)
        k: torch.Tensor = self.k_proj(key)
        v: torch.Tensor = self.v_proj(value)

        q = q.reshape(B, Q, self.n_heads, D_MODEL//self.n_heads).transpose(1, 2) # B, N_HEADS, Q, HEAD_DIM
        k = k.reshape(B, K, self.n_heads, D_MODEL//self.n_heads).transpose(1, 2)
        v = v.reshape(B, K, self.n_heads, D_MODEL//self.n_heads).transpose(1, 2)

        if self.flash:
            attn_weights = F.scaled_dot_product_attention(query=q,
                                                          key=k,
                                                          value=v,
                                                          is_causal=self.is_causal,
                                                          dropout_p=self.dropout_p)
        else:

            attn_weights = q @ k.transpose(-2, -1)/math.sqrt(D_MODEL)

            if self.is_causal:
                mask = torch.ones(size=(1, 1, Q, K)).tril(diagonal=0)
                attn_weights = attn_weights.masked_fill(mask=mask.logical_not(), value=float("-inf"))
            
            attn_weights = F.softmax(input=attn_weights, dim=-1)
            attn_weights = self.dropout(attn_weights) @ v
        
        # attn_weights -> B, N_HEADS, Q, HEAD_DIM
        attn_weights = attn_weights.transpose(1, 2).contiguous().view(B, Q, D_MODEL)
        return self.o_proj(attn_weights)
    

class FFN(nn.Module):

    def __init__(self, d_model: int, dropout_p: float=0.1):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.GELU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(d_model*4, d_model)
        )
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderBlock(nn.Module):

    def __init__(self, n_heads: int, d_model: int, dropout_p: float=0.1) -> None:

        super().__init__()
        
        self.layer_norm = nn.LayerNorm(normalized_shape=d_model)
        self.mha = MHA(n_heads=n_heads, d_model=d_model, dropout_p=dropout_p, is_causal=False)
        self.ffn = FFN(d_model=d_model, dropout_p=dropout_p)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:

        # x-> B, Q, D_MODEL
        x = self.layer_norm(x + self.mha(x, x, x))
        x = self.layer_norm(x + self.ffn(x))

        return x


class Encoder(nn.Module):

    def __init__(self, n_layers: int, n_heads: int, d_model: int, dropout_p: float=0.1) -> None:

        super().__init__()
        self.blocks = nn.ModuleDict([EncoderBlock(n_heads, d_model, dropout_p) for _ in range(n_layers)])
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:

        # x -> B, Q, D_MODEL
        for block in self.blocks:
            x = block(x)
        
        return x
    
class DecoderBlock(nn.Module):

    def __init__(self, n_heads: int, d_model: int, dropout_p: float=0.1) -> None:

        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=d_model)
        self.masked_mha = MHA(n_heads=n_heads, d_model=d_model, dropout_p=dropout_p, is_causal=True)
        self.cross_mha = MHA(n_heads=n_heads, d_model=d_model, dropout_p=dropout_p, is_causal=False)
        self.ffn = FFN(d_model=d_model, dropout_p=dropout_p)
    
    def forward(self, encoder_out: torch.Tensor, decoder_in: torch.Tensor) -> torch.Tensor:
        
        masked_out = self.layer_norm(decoder_in + self.masked_mha(decoder_in, decoder_in, decoder_in))
        cross_out = self.layer_norm(masked_out + self.cross_mha(masked_out, encoder_out, encoder_out))
        ffn_out = self.layer_norm(cross_out + self.ffn(cross_out))
        return ffn_out

class Decoder(nn.Module):

    def __init__(self, n_layers: int, n_heads: int, d_model: int, dropout_p: float=0.1) -> None:

        super().__init__()

        self.blocks = nn.ModuleList([DecoderBlock(n_heads, d_model, dropout_p) for _ in range(n_layers)])
    
    def forward(self, encoder_out: torch.Tensor, decoder_in: torch.Tensor) -> torch.Tensor:

        for block in self.blocks:
            decoder_in = block(encoder_out, decoder_in)
        
        return decoder_in
    

class TranslateFormer(nn.Module):

    def __init__(self, 
                 input_vocab_size: int,
                 output_vocab_size: int,
                 max_seq_len: int,
                 n_layers: int, 
                 n_heads: int, 
                 d_model: int, 
                 dropout_p: float=0.1) -> None:

        super().__init__()
        self.input_embedding = Embedding(input_vocab_size, d_model, max_seq_len, dropout_p)
        self.output_embedding = Embedding(output_vocab_size, d_model, max_seq_len, dropout_p)
        self.encoder = Encoder(n_layers, n_heads, d_model, dropout_p)
        self.decoder = Decoder(n_layers, n_heads, d_model, dropout_p)
        self.cls_net = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(in_features=d_model, out_features=output_vocab_size)
        )

    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

        encoder_in: torch.Tensor = self.input_embedding(x)
        decoder_in: torch.Tensor = self.output_embedding(y)

        encoder_out: torch.Tensor = self.encoder(encoder_in)
        decoder_out: torch.Tensor = self.decoder(encoder_out, decoder_in)

        logits: torch.Tensor = self.cls_net(decoder_out) # B, K, OUTPUT_VOCAB_SIZE

        B, SEQ_LEN, _ = logits.shape

        loss = F.cross_entropy(logits.reshape(B*SEQ_LEN, -1), target=y.reshape(B*SEQ_LEN,),
                               ignore_index=Tokens.pad_token)
        return loss