import torch
from torch import nn
import torch.nn.functional as F
import math


class CausalSelfAttention(nn.Module):

    def __init__(self, config) -> None:

        super().__init__()
        d_model = config['d_model']
        self.head_dim = config['head_dim']
        self.n_heads = config['n_heads']
        self.qkv_proj = nn.Linear(d_model, d_model*3)
        self.o_proj = nn.Linear(d_model, d_model)
        self.drop_layer = nn.Dropout(p=config['attn_dropout'])
        
        ctx_length = config['ctx_length']
        mask = torch.tril(input=torch.ones((ctx_length, ctx_length), requires_grad=False)).unsqueeze(0).unsqueeze(1)
        self.register_buffer('mask', mask, persistent=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        b, t, d_model = x.shape
        q, k, v = self.qkv_proj(x).split(d_model, 2) # b, t, d_model
        q = q.reshape(b, t, self.n_heads, self.head_dim).transpose(1, 2) # b, n_heads, t, head_dim
        k = k.reshape(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        attn_wts = (q @ k.transpose(-1, -2)) / math.sqrt(d_model) # b, n_heads, t, t
        attn_wts = attn_wts.masked_fill(self.mask[:, :, :t, :t]==0, value=float("-inf"))
        attn_wts = F.softmax(self.drop_layer(attn_wts), dim=-1)
        y = attn_wts @ v # b, n_heads, t, head_dim
        y = y.transpose(1, 2).contiguous().view(b, t, -1)
        return self.o_proj(y)


class MLP(nn.Module):

    def __init__(self, config) -> None:

        super().__init__()
        d_model = config['d_model']
        self.seq_layer = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.GELU(),
            nn.Dropout(p=config['mlp_dropout']),
            nn.Linear(d_model*4, d_model)
        )
    
    def forward(self, x: torch.Tensor): return self.seq_layer(x)


class DecoderBlock(nn.Module):

    def __init__(self, config) -> None:

        super().__init__()
        self.mha = CausalSelfAttention(config)
        self.ln_1 = nn.LayerNorm(config['d_model'])
        self.mlp = MLP(config)
        self.ln_2 = nn.LayerNorm(config['d_model'])
    
    def forward(self, x: torch.Tensor):
        x = x + self.mha(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config):

        super().__init__()
        self.decoder = nn.ModuleDict(dict(
            wte = nn.Embedding(config['vocab_size'], config['d_model']),
            wpe = nn.Embedding(config['ctx_length'], config['d_model']),
            blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config['n_decoders'])]),
            lm_head = nn.Linear(config['d_model'], config['vocab_size'])
        ))

        # tie the weights to reduce params
        self.decoder.wte.weight = self.decoder.lm_head.weight
        self.apply(self._init_weights)

    def forward(self, x, y=None):
        b, t = x.shape
        tok_emb = self.decoder.wte(x) # b, t, d_model
        pos_emb = self.decoder.wpe(torch.arange(0, t, device=x.device).unsqueeze(0)) # 1, t, d_model
        x = tok_emb + pos_emb
        for block in self.decoder.blocks:
            x = block(x)

        logits = self.decoder.lm_head(x) # b, t, vocab_size    
        loss = None
        if y is not None:
            # y -> b, t
            loss = F.cross_entropy(logits.view(b*t, -1), y.view(-1))
        return logits, loss
    

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            nn.init.uniform_(m.weight, -1/math.sqrt(m.embedding_dim), 1/math.sqrt(m.embedding_dim))
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class pGPT(nn.Module):

    def __init__(self, config):

        super().__init__()
        self.d_model = config['d_model']
        self.decoder = nn.ModuleDict(dict(
            wte = nn.Embedding(config['vocab_size'], config['d_model']*2),
            wpe = nn.ModuleDict(config['ctx_length'], config['d_model']*2),
            path_1 = nn.ModuleList([DecoderBlock(config) for _ in range(config['n_decoders']//2)]),
            path_2 = nn.ModuleList([DecoderBlock(config) for _ in range(config['n_decoders']//2)]),
            lm_head = nn.Linear(config['d_model'], config['vocab_size']),
            path_weight = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        ))
        
        self.apply(self._init_weights)
    
    def forward(self, x, y=None):
        b, t = x.shape
        tok_emb1, tok_emb2 = self.decoder.wte(x).split(self.d_model, 2)
        pos_emb1, pos_emb2 = self.decoder.wpe(torch.arange(0, t, device=x.device).unsqueeze(0)).split(self.d_model, 2)
        x1, x2 = tok_emb1 + pos_emb1, tok_emb2 + pos_emb2

        for block1, block2 in zip(self.decoder.path_1, self.decoder.path_2):
            x1 = block1(x1)
            x2 = block2(x2)
        x = self.decoder.path_weight*x1 + (1-self.decoder.path_weight)*x2
        logits = self.decoder.lm_head(x)
        
        loss = None
        if y is not None:
            # y -> b, t
            loss = F.cross_entropy(logits.view(b*t, -1), y.view(-1))
        
        return logits, loss
    

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            nn.init.uniform_(m.weight, -1/math.sqrt(m.embedding_dim), 1/math.sqrt(m.embedding_dim))
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class ConvDecoderBlock(nn.Module):

    def __init__(self, config):

        super().__init__()
        self.conv_block = nn.ModuleDict(dict(
            block_1 = DecoderBlock(config),
            block_2 = DecoderBlock(config),
            conv1 = nn.Conv1d(config["d_model"], config["d_model"]//2, 1)
        ))
    
    def forward(self, x):

        x = self.conv_block.block_1(x)
        x = self.conv_block.block_2(x).transpose(1, 2) # b, d_model, t
        x = self.conv_block.conv1(x).transpose(1, 2) # b, t, d_model//2
        return x
    

class ccGPT(nn.Module):

    def __init__(self, config):

        super().__init__()
        final_dim = config['d_model'] // (2**(config['n_layers']//2 - 1))
        self.decoder = nn.ModuleDict(dict(
            wte = nn.Embedding(config['vocab_size'], config['d_model']),
            wpe = nn.Embedding(config['ctx_length'], config['d_model']),
            blocks = nn.ModuleList([]),
            lm_head = nn.Linear(final_dim, config['vocab_size'])
        ))

        for _ in range(config['n_layers']//2):
            self.decoder.blocks.append(ConvDecoderBlock(config))
            config["d_model"] //= 2
    
        self.apply(self._init_weights)

    def forward(self, x, y=None):

        b, t = x.shape
        tok_emb = self.decoder.wte(x)
        pos_emb = self.decoder.wpe(torch.arange(0, t, device=x.device).unsqueeze(0))
        x = tok_emb + pos_emb
        for block in self.decoder.blocks:
            x = block(x)
        
        logits = self.block.lm_head(x)
        loss = None
        if y is not None:
            # y -> b, t
            loss = F.cross_entropy(logits.view(b*t, -1), y.view(-1))
        
        return logits, loss
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            nn.init.uniform_(m.weight, -1/math.sqrt(m.embedding_dim), 1/math.sqrt(m.embedding_dim))
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class lcGPT(nn.Module):

    def __init__(self, config):

        super().__init__()
        final_dim = config['d_model'] // (2**(config['n_layers']//2 - 1))
        self.decoder = nn.ModuleDict(dict(
            wte = nn.Embedding(config['vocab_size'], config['d_model']),
            wpe = nn.Embedding(config['ctx_length'], config['d_model']),
            blocks = nn.ModuleList([]),
            lm_head = nn.Linear(final_dim, config['vocab_size'])
        ))

        for _ in range(config['n_layers']//2):
            self.decoder.blocks.extend([
                DecoderBlock(config),
                DecoderBlock(config),
                nn.Linear(config['d_model'], config['d_model']//2)
            ])
            config["d_model"] //= 2

        self.apply(self._init_weights)

    def forward(self, x, y=None):

        b, t = x.shape
        tok_emb = self.decoder.wte(x)
        pos_emb = self.decoder.wpe(torch.arange(0, t, device=x.device).unsqueeze(0))
        x = tok_emb + pos_emb
        for block in self.decoder.blocks:
            x = block(x)
        
        logits = self.block.lm_head(x)
        loss = None
        if y is not None:
            # y -> b, t
            loss = F.cross_entropy(logits.view(b*t, -1), y.view(-1))
        
        return logits, loss

   
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            nn.init.uniform_(m.weight, -1/math.sqrt(m.embedding_dim), 1/math.sqrt(m.embedding_dim))
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)