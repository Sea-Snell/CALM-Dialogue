from typing import Any, List, Set, Tuple, Callable, Union
import torch
import torch.nn as nn
import numpy as np

def map_pytree(f: Callable[[Union[np.ndarray, torch.Tensor]], Any], 
               item: Any):
    if isinstance(item, dict):
        return {k: map_pytree(f, v) for k, v in item.items()}
    elif isinstance(item, List) or isinstance(item, Set) or isinstance(item, Tuple):
        return [map_pytree(f, v) for v in item]
    elif isinstance(item, np.ndarray) or isinstance(item, torch.Tensor):
        return f(item)
    else:
        return item

def to(item: Any, device: torch.device):
    return map_pytree(lambda x: torch.tensor(x).to(device), item)

def to_decorator(f, device):
    def new_f(*args, **kwargs):
        return to(f(*args, **kwargs), device)
    return new_f

def parameter_norm(model: nn.Module):
    norm = 0.0
    for param in model.parameters():
        norm += param.norm().item()
    return norm

def get_transformer_logs(attentions: List[torch.Tensor], model: nn.Module, attn_mask: torch.Tensor):
    logs = {}
    n = attn_mask.sum()
    model_attention_entropy = -sum(map(lambda x: ((x * torch.log(x+1e-5)).sum(dim=-1) * attn_mask.unsqueeze(1)).sum().item(), attentions)) / (len(attentions) * n)
    model_parameter_norm = parameter_norm(model)
    logs['attention_entropy'] = (model_attention_entropy, n * len(attentions))
    logs['parameter_norm'] = (model_parameter_norm, 1)
    return logs

