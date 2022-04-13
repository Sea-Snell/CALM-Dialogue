from typing import List
import torch

def kl_divergence(a: torch.Tensor, b: torch.Tensor):
    # a = (b, d)
    # b = (b, d)
    return ((a * torch.log(a+1e-5)) - (a * torch.log(b+1e-5))).sum(dim=1)

def uniform_prior(attn_mask: torch.Tensor):
    return attn_mask * (1 / attn_mask.sum(dim=1)).unsqueeze(dim=1)

def geometric_prior(attn_mask: torch.Tensor, rate: float):
    return attn_mask * ((rate ** (torch.flip(torch.cumsum(torch.flip(attn_mask, [1]), 1)-1, [1]))) * (1 - rate)) / (1 - rate**(attn_mask.sum(dim=1))).unsqueeze(1)

