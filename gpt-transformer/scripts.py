import torch
from torch import nn
import torch.nn.functional as F
import math
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Embedding(nn.Module):
    """
    Embedding module for the tokens
    """

    def __init__(self, config):
        """
        Initializes the Embedding module.

        Args:
            config (dict): Configuration dictionary containing `vocab_size`, `d_model`, 
                           `context_length`, `dropout`, and `device`.
        """
        super().__init__()

        # Initialize module parameters from the configuration dictionary
        self.vocab_size = config["vocab_size"]
        self.d_model = config["d_model"]
        self.context_length = config["context_length"]
        self.device = config["device"]

        # Embedding layers
        self.token_embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.d_model)
        self.pos_embedding = nn.Embedding(num_embeddings=self.context_length, embedding_dim=self.d_model)

        # Dropout layer
        self.dropout = nn.Dropout(p=config["dropout"])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the module.

        Args:
            x (torch.Tensor): The input tensor containing token indices. Shape [B, T] where
                              B is batch size and T is sequence length.

        Returns:
            torch.Tensor: The resulting tensor after applying embeddings and dropout. Shape [B, T, D_MODEL].

        Raises:
            AssertionError: If the sequence length of the input exceeds `context_length`.
        """
        B, T = x.shape  # Batch size (B) and sequence length (T)

        # Ensure the sequence length does not exceed the maximum allowed length
        assert T <= self.context_length, AssertionError(f"Sequence length {T} should be less than or equal to {self.context_length}")

        # Create a tensor of positions [0, 1, ..., T-1], and expand to match batch size
        position = torch.arange(start=0, end=T, dtype=torch.int64).unsqueeze(dim=0).to(self.device)

        # Compute token and position embeddings
        tok_emb = self.token_embedding(x)         # Token embeddings [B, T, D_MODEL]
        pos_emb = self.pos_embedding(position)    # Position embeddings [1, T, D_MODEL]

        # Sum token and position embeddings and apply dropout
        return self.dropout(tok_emb + pos_emb)

    
class MHA(nn.Module):

    """
    Masked Multi-Head Attention Block for the decoder
    """

    def __init__(self, config) -> None:
        
        """
        Initializes the MHA block
        Args:
            config (dict)
        """

        super().__init__()

        assert config["d_model"]%config["n_heads"]==0, AssertionError(f"d_model: {config['d_model']} should be divisible by n_heads: {config['n_heads']}")

        self.n_heads = config["n_heads"]
        self.d_model = config["d_model"]
        self.head_dim = self.d_model//self.n_heads
        self.dropout_p = config["dropout"]
        
        self.proj = nn.Linear(in_features=self.d_model, out_features=self.d_model*3)
        self.o_proj = nn.Linear(in_features=self.d_model, out_features=self.d_model)
        self.dropout = nn.Dropout(p=self.dropout_p)
        # create and register the causal mask
        mask = torch.ones(size=(1, 1, config["context_length"], config["context_length"]), dtype=torch.bool).tril(diagonal=0)
        self.register_buffer(name="mask", tensor=mask)


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """
        Defines the forward pass of the MHA bloc
        Args:
            x (torch.Tensor)
        """

        B, T, D_MODEL = x.shape
        # get query, key, value from the projection of x from D_MODEL to D_MODEL*3
        q, k, v = self.proj(x).split(D_MODEL, dim=2) # B, T, D_MODEL
        # reshape q,k,v for calculating self-attention
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2) 
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        attn_outputs: torch.Tensor = (q @ k.transpose(-2, -1))*(1/math.sqrt(k.size(-1)))
        attn_outputs = attn_outputs.masked_fill(self.mask[:, :, :T, :T]==0, value=float("-inf")) # fill in the mask
        attn_outputs = F.softmax(attn_outputs, dim=-1)
        attn_outputs = self.dropout(attn_outputs) @ v
        
        attn_outputs = attn_outputs.transpose(1, 2).contiguous().view(B, T, D_MODEL)
        return self.o_proj(attn_outputs)
    

class FFN(nn.Module):

    """
    Fully connected feed forward block :- sub-block the decoder block
    """

    def __init__(self, config):

        """
        Initializes the feed forward block
        Args:
            config (dict)
        """

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
        self.layer_norm_1 = nn.LayerNorm(normalized_shape=config["d_model"])
        self.layer_norm_2 = nn.LayerNorm(normalized_shape=config["d_model"])
        self.masked_mha = MHA(config)
        self.ffn = FFN(config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        masked_out = self.layer_norm_1(x + self.masked_mha(x))
        ffn_out = self.layer_norm_2(masked_out + self.ffn(masked_out))
        return ffn_out


class Decoder(nn.Module):

    def __init__(self, config) -> None:

        super().__init__()
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config["n_layers"])])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        for block in self.blocks:
            x = block(x)
        
        return x
    

class LCDecoder(nn.Module):

    def __init__(self, config) -> None:

        super().__init__()

        self.blocks = nn.ModuleList()
        for _ in range(config["n_layers"]//2):
            self.blocks.extend([
                DecoderBlock(config),
                DecoderBlock(config),
                nn.Linear(config["d_model"], config["d_model"]//2)
            ])
            config["d_model"] //= 2
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        for block in self.blocks:
            x = block(x)
        
        return x
    

class ConvCompressLayer(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int=1) -> None:

        super().__init__()

        self.compress_layer = nn.Conv1d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = x.transpose(1, 2)
        x = self.compress_layer(x)
        x = x.transpose(1, 2)
        return x
    

class CCDecoder(nn.Module):

    def __init__(self, config) -> None:

        super().__init__()

        self.blocks = nn.ModuleList()
        for _ in range(config["n_layers"]//2):
            self.blocks.extend([
                DecoderBlock(config),
                DecoderBlock(config),
                ConvCompressLayer(config["d_model"], config["d_model"]//2)
            ])
            config["d_model"] //= 2
            
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