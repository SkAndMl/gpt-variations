## code adapted from "https://github.com/chaoluond/quicktypeGPT/blob/main/model.py"

import torch
from torch import nn
from typing import Tuple


class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float) -> None:

        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x: torch.Tensor): return x*torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor): return self.weight*self._norm(x.float()).type_as(x)


## adding functions for implementing RoPE
def precompute_freqs_cis(dim: int, end: int, theta: float=10000.0) -> Tuple[torch.Tensor]:

   freqs = 1.0 / (theta**(torch.arange(0, dim, 2)[: dim//2].float() / dim)) 
   t = torch.arange(end, device=freqs.device)
   freqs = torch.outer(t, freqs).float()
   freqs_cos = torch.cos(freqs)
   freqs_sin = torch.sin(freqs)
   return freqs_cos, freqs_sin


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:

    ndim = x.ndim
    assert freqs_cis.shape == (x.shape[-1], x.shape[-1])
    shape = [d if i==ndim-2 or i==ndim-1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)