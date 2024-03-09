import torch
from torch import nn
import torch.nn.functional as F
import math

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
        self.weight = nn.Parameter(data=0.5, requires_grad=True)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:

        # x-> B, Q, D_MODEL
        mha_out = x + self.mha(x, x, x)
        ffn_out = x + self.ffn(x)

        return self.weight*mha_out + (1-self.weight)*ffn_out


class Encoder(nn.Module):

    def __init__(self, n_layers: int, n_heads: int, d_model: int, dropout_p: float=0.1) -> None:

        super().__init__()
        self.blocks = nn.ModuleDict([EncoderBlock(n_heads, d_model, dropout_p) for _ in range(n_layers)])
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:

        # x -> B, Q, D_MODEL
        for block in self.blocks:
            x = block(x)
        
        return x