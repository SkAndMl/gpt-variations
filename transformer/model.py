import torch
from torch import nn
import torch.nn.functional as F
import math

class Tokens:

    pad_token = 0
    sos_token = 1
    eos_token = 2
    unk_token = 3

    pad = "<pad>"
    sos = "<sos>"
    eos = "<eos>"
    unk = "<unk>"

class Embedding(nn.Module):

    def __init__(self, config, vocab_size: int) -> None:

        super().__init__()

        self.config = config
        self.token_embedding = nn.Embedding(num_embeddings=vocab_size,
                                            embedding_dim=config["d_model"])
        self.pos_embedding = nn.Embedding(num_embeddings=config["seq_len"],
                                          embedding_dim=config["d_model"])
        self.dropout = nn.Dropout(p=config["dropout"])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # x -> B, SEQ_LEN
        _, SEQ_LEN = x.shape
        pos_ids = torch.arange(0, SEQ_LEN, device=self.config["device"], dtype=torch.long).unsqueeze(0) # 1, SEQ_LEN

        tok_embeds = self.token_embedding(x) # B, SEQ_LEN, D_MODEL
        pos_embeds = self.pos_embedding(pos_ids) # 1, SEQ_LEN, D_MODEL

        return self.dropout(tok_embeds + pos_embeds)


class MHA(nn.Module):

    def __init__(self, config) -> None:
        
        super().__init__()

        assert config["d_model"]%config["n_heads"] == 0

        self.config = config
        self.q_proj = nn.Linear(config["d_model"], config["d_model"])
        self.k_proj = nn.Linear(config["d_model"], config["d_model"])
        self.v_proj = nn.Linear(config["d_model"], config["d_model"])
        self.o_proj = nn.Linear(config["d_model"], config["d_model"])
        self.dropout = nn.Dropout(p=config["dropout"])
    
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor, 
                mask: torch.Tensor=None) -> torch.Tensor:
        
        B, SEQ_LEN, D_MODEL = query.shape
        head_dim = D_MODEL//self.config["n_heads"]

        q: torch.Tensor = self.q_proj(query)
        k: torch.Tensor = self.k_proj(key)
        v: torch.Tensor = self.v_proj(value)

        q = q.view(B, SEQ_LEN, self.config["n_heads"], head_dim).transpose(1, 2) # B, N_HEADS, SEQ_LEN, HEAD_DIM
        k = k.view(B, SEQ_LEN, self.config["n_heads"], head_dim).transpose(1, 2)
        v = v.view(B, SEQ_LEN, self.config["n_heads"], head_dim).transpose(1, 2)

        attn_weights = q@k.transpose(-1, -2) # B, N_HEADS, SEQ_LEN, SEQ_LEN
        attn_weights /= math.sqrt(head_dim)
        
        if mask is not None:
            attn_weights.masked_fill_(mask==0, value=float("-inf"))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)@v # B, N_HEADS, SEQ_LEN, HEAD_DIM

        attn_weights = attn_weights.transpose(1, 2).contiguous().view(B, SEQ_LEN, D_MODEL)
        return self.o_proj(attn_weights)


class FFN(nn.Module):

    def __init__(self, config) -> None:
        
        super().__init__()

        self.ffn = nn.Sequential(
            nn.Linear(config["d_model"], config["d_model"]*4),
            nn.GELU(),
            nn.Dropout(p=config["dropout"]),
            nn.Linear(config["d_model"]*4, config["d_model"])
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.ffn(x)


class EncoderBlock(nn.Module):

    def __init__(self, config):

        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=config["d_model"])
        self.mha_block = MHA(config=config)
        self.ffn_block = FFN(config=config)
    
    def forward(self, x:torch.Tensor, mask:torch.Tensor=None) -> torch.Tensor:

        x = self.layer_norm(x + self.mha_block(x, x, x, mask))
        x = self.layer_norm(x + self.ffn_block(x))

        return x

class Encoder(nn.Module):

    def __init__(self, config) -> None:

        super().__init__()

        self.encoder_blocks = nn.ModuleList([EncoderBlock(config) for _ in range(config["n_layers"])])
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:

        for block in self.encoder_blocks:
            x = block(x, mask)
        
        return x


class DecoderBlock(nn.Module):

    def __init__(self, config) -> None:

        super().__init__()

        self.layer_norm = nn.LayerNorm(normalized_shape=config["d_model"])
        self.masked_mha = MHA(config=config)
        self.cross_mha = MHA(config=config)
        self.ffn = FFN(config=config)
    
    def forward(self, 
                encoder_output: torch.Tensor, 
                decoder_input: torch.Tensor, 
                encoder_mask: torch.Tensor=None,
                decoder_mask: torch.Tensor=None) -> torch.Tensor:
        

        masked_out = self.layer_norm(decoder_input + self.masked_mha(decoder_input, decoder_input, decoder_input, decoder_mask))
        cross_out = self.layer_norm(masked_out + self.cross_mha(masked_out, encoder_output, encoder_output, encoder_mask))
        ffn_out = self.layer_norm(cross_out + self.ffn(cross_out))

        return ffn_out
    
class Decoder(nn.Module):

    def __init__(self, config) -> None:

        super().__init__()

        self.decoder_blocks = nn.ModuleList([DecoderBlock(config=config) for _ in range(config["n_layers"])])
    
    def forward(self,
                encoder_output: torch.Tensor,
                decoder_input: torch.Tensor,
                encoder_mask: torch.Tensor,
                decoder_mask: torch.Tensor) -> torch.Tensor:
        
        for block in self.decoder_blocks:
            decoder_input = block(encoder_output, decoder_input, encoder_mask, decoder_mask)
        
        return decoder_input


class TranslateFormer(nn.Module):

    def __init__(self, config) -> None:
        
        super().__init__()

        self.encoder_embedding = Embedding(config=config, vocab_size=config["input_vocab_size"])
        self.decoder_embedding = Embedding(config=config, vocab_size=config["output_vocab_size"])
        self.encoder = Encoder(config=config)
        self.decoder = Decoder(config=config)
        self.cls_layer = nn.Linear(config["d_model"], config["output_vocab_size"])
    
    def forward(self,
                encoder_input: torch.Tensor,
                decoder_input: torch.Tensor,
                encoder_mask: torch.Tensor,
                decoder_mask: torch.Tensor) -> torch.Tensor:
        
        # encoder_input -> B, SEQ_LEN
        # decoder_input -> B, SEQ_LEN

        encoder_input = self.encoder_embedding(encoder_input) # B, SEQ_LEN, D_MODEL
        decoder_input = self.decoder_embedding(decoder_input) # B, SEQ_LEN, D_MODEL

        encoder_output: torch.Tensor = self.encoder(encoder_input, encoder_mask)
        decoder_output: torch.Tensor = self.decoder(encoder_output, decoder_input, encoder_mask, decoder_mask)

        logits: torch.Tensor = self.cls_layer(decoder_output)
        return logits