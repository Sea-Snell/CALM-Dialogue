import functools
from logging import log
import math
from typing import Dict, Any, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from ad.airdialogue import AgentScenario
from models.auxilary_utils import geometric_prior, kl_divergence, uniform_prior
from models.embedding_modules import EmbeddingCombiner
from utils.data_utils import ActionEnumerator, DiscreteFeatures
from utils.misc import stack_dicts


class Aux2(nn.Module):
    def __init__(self, emb_dim, device):
        super(Aux2, self).__init__()
        self.emb_dim = emb_dim
        self.status_head = nn.Linear(emb_dim, 5)
        self.flight_head = nn.Linear(emb_dim, 30)
        self.a_enum= ActionEnumerator()
        self.device = device

    def forward(self, embs: torch.Tensor):
        # embs = (b, t, d)
        status_probs = F.softmax(self.status_head(embs), dim=-1)
        no_flight, no_reservation, cancel, change, book = status_probs[:, :, 0], status_probs[:, :, 1],\
                                                          status_probs[:, :, 2], status_probs[:, :, 3],\
                                                          status_probs[:, :, 4]
        flight_probs = F.softmax(self.flight_head(embs), dim=-1)

        bsize, t = embs.shape[:2]
        final_probs = torch.empty((bsize, t, self.a_enum.n_actions(),)).to(self.device)
        final_probs[:, :, self.a_enum.action2idx({'status': 'no_flight', 'flight': []})] = no_flight
        final_probs[:, :, self.a_enum.action2idx({'status': 'cancel', 'flight': []})] = cancel
        final_probs[:, :, self.a_enum.action2idx({'status': 'no_reservation', 'flight': []})] = no_reservation
        for i in range(0, 30):
            final_probs[:, :, self.a_enum.action2idx({'status': 'book', 'flight': [i+1000]})] = book * flight_probs[:, :, i]
            final_probs[:, :, self.a_enum.action2idx({'status': 'change', 'flight': [i+1000]})] = change * flight_probs[:, :, i]
        return final_probs
    
    def get_loss(self, embs: torch.Tensor, attn_mask: torch.Tensor, expected_actions: List[Dict[str, Any]]):
        # embs = (b, t, d)
        logs = {}
        assert embs.shape[0] == len(expected_actions)
        output_probs = self(embs)
        bsize, t = output_probs.shape[:2]
        valid_probs = torch.zeros((bsize,t,)).to(self.device)
        for i in range(bsize):
            valid_actions = self.a_enum.factor_expected_action(expected_actions[i])
            for action in valid_actions:
                valid_probs[i, :] += output_probs[i, :, self.a_enum.action2idx(action)]
        n = attn_mask.sum()
        loss = -(torch.log(valid_probs+1e-5) * attn_mask).sum() / n
        acc = (valid_probs * attn_mask).sum() / n
        acc_max = torch.max(valid_probs.masked_fill_(attn_mask == 0, float('-inf')), dim=1).values.mean()
        logs['loss'] = (loss.item(), n.item())
        logs['acc'] = (acc.item(), n.item())
        logs['acc_max'] = (acc_max.item(), bsize)
        return loss, logs, []





