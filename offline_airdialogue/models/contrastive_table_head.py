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

class ContrastiveTableHead(nn.Module):
    def __init__(self, emb_dim, device):
        super().__init__()
        self.emb_dim = emb_dim
        self.goal_head = nn.Linear(emb_dim, 3)
        self.book_head = nn.Linear(emb_dim, emb_dim)
        self.change_head = nn.Linear(emb_dim, emb_dim)
        self.cancel_head = nn.Linear(emb_dim, emb_dim)
        self.change_gate = nn.Linear(2*emb_dim, 1)
        self.cancel_gate = nn.Linear(2*emb_dim, 1)
        self.a_enum= ActionEnumerator()
        self.device = device
    
    def _goal_probs(self, query_embs: torch.Tensor):
        goals = F.softmax(self.goal_head(query_embs), dim=1)
        return goals[:, 0], goals[:, 1], goals[:, 2]

    def _get_flight_probs(self, query_embs: torch.Tensor, table_embs: torch.Tensor, no_flight_emb: torch.Tensor):
        no_flight_table = torch.cat((no_flight_emb.unsqueeze(1), table_embs,), dim=1)
        probs = F.softmax(torch.einsum('brd,bd->br', no_flight_table, query_embs) / math.sqrt(self.emb_dim), dim=1)
        no_flight_prob, table_probs = probs[:, 0], probs[:, 1:]
        return no_flight_prob, table_probs
    
    def _change_prob(self, query_embs: torch.Tensor, res_emb: torch.Tensor):
        return torch.sigmoid(self.change_gate(torch.cat((query_embs, res_emb), dim=1)).squeeze(1))
    
    def _cancel_prob(self, query_embs: torch.Tensor, res_emb: torch.Tensor):
        return torch.sigmoid(self.cancel_gate(torch.cat((query_embs, res_emb), dim=1)).squeeze(1))

    def forward(self, constraint_embs: torch.Tensor, table_embs: torch.Tensor, no_flight_emb: torch.Tensor, res_emb: torch.Tensor):
        # constraint_embs = (b, d)
        # table_embs = (b, r, d)
        # no_flight_emb = (b, d)
        # res_emb = (b, d)
        book_goal_prob, change_goal_prob, cancel_goal_prob = self._goal_probs(constraint_embs)
        book_emb = self.book_head(constraint_embs)
        change_emb = self.change_head(constraint_embs)
        cancel_emb = self.cancel_head(constraint_embs)
        book_no_flight, book_table = self._get_flight_probs(book_emb, table_embs, no_flight_emb)
        change_prob = self._change_prob(change_emb, res_emb)
        change_no_flight, change_table = self._get_flight_probs(change_emb, table_embs, no_flight_emb)
        cancel_prob = self._cancel_prob(cancel_emb, res_emb)

        bsize = constraint_embs.shape[0]
        final_probs = torch.empty((bsize, self.a_enum.n_actions(),)).to(self.device)
        final_probs[:, self.a_enum.action2idx({'status': 'no_flight', 'flight': []})] = \
            book_goal_prob * book_no_flight + change_goal_prob * change_prob * change_no_flight
        final_probs[:, self.a_enum.action2idx({'status': 'cancel', 'flight': []})] = \
            cancel_goal_prob * cancel_prob
        final_probs[:, self.a_enum.action2idx({'status': 'no_reservation', 'flight': []})] = \
            cancel_goal_prob * (1-cancel_prob) + change_goal_prob * (1-change_prob)
        for i in range(0, 30):
            final_probs[:, self.a_enum.action2idx({'status': 'book', 'flight': [i+1000]})] = \
                book_goal_prob * book_table[:, i]
            final_probs[:, self.a_enum.action2idx({'status': 'change', 'flight': [i+1000]})] = \
                change_goal_prob * change_prob * change_table[:, i]
        return final_probs
    
    def get_loss(self, output_probs: torch.Tensor, expected_actions: List[Dict[str, Any]]):
        # output_probs = (b, d)
        logs = {}
        assert output_probs.shape[0] == len(expected_actions)
        bsize = output_probs.shape[0]
        valid_probs = torch.zeros((bsize,)).to(self.device)
        for i in range(bsize):
            valid_actions = self.a_enum.factor_expected_action(expected_actions[i])
            for action in valid_actions:
                valid_probs[i] += output_probs[i, self.a_enum.action2idx(action)]
        loss = -torch.log(valid_probs+1e-5).mean()
        acc = valid_probs.mean()
        logs['loss'] = (loss.item(), bsize)
        logs['acc'] = (acc.item(), bsize)
        return loss, logs, []
        
class TransformerTableHead(nn.Module):
    def __init__(self, 
                 discrete_features: DiscreteFeatures, 
                 emb_dim: int, 
                 device: Union[torch.device, str] = "cuda", 
                 attn_prior: str = 'geometric', 
                 geometric_rate: float = 0.9):
        super(TransformerTableHead, self).__init__()
        self.discrete_features = discrete_features
        self.emb_dim = emb_dim
        self.flight_encoder = EmbeddingCombiner(self.discrete_features.get_emb_spec(), 
                                                self.emb_dim)
        self.contrastive_table_head = ContrastiveTableHead(self.emb_dim, device)
        self.hidden_attn_proj = nn.Linear(self.emb_dim, 1)
        if attn_prior == 'geometric':
            self.attn_prior = functools.partial(geometric_prior, 
                                                rate=geometric_rate)
        elif attn_prior == 'uniform':
            self.attn_prior = uniform_prior
        else:
            raise NotImplementedError
        self.device = device
    
    def _tokenize_agent_scenarios_batch(self, scenarios: List[AgentScenario]):
        return {k: torch.tensor(v).to(self.device) for k, v in stack_dicts([scenario.get_discrete_state(self.discrete_features) 
                                                                            for scenario in scenarios]).items()}
    
    def embed_agent_context(self, agent_states: List[AgentScenario]):
        flight_tables = self._tokenize_agent_scenarios_batch(agent_states)
        embeddings = self.flight_encoder(flight_tables)
        return embeddings
    
    def forward(self, table_embs: torch.Tensor, 
                no_flight_emb: torch.Tensor, res_emb: torch.Tensor,
                token_states: torch.Tensor, attn_mask: torch.Tensor):
        hidden_attn = F.softmax(self.hidden_attn_proj(token_states).squeeze(2).masked_fill_(attn_mask==0, float('-inf')), dim=1)
        collapsed_hidden_state = torch.einsum('btd,bt->bd', token_states, hidden_attn)
        attn_kl = kl_divergence(hidden_attn, self.attn_prior(attn_mask)).mean()
        contrastive_predictions = self.contrastive_table_head(collapsed_hidden_state, 
                                                              table_embs, 
                                                              no_flight_emb, 
                                                              res_emb)
        return attn_kl, contrastive_predictions
    
    def get_loss(self, table_embs_states: torch.Tensor, 
                 no_flight_emb: torch.Tensor, res_emb: torch.Tensor, 
                 token_states: torch.Tensor, attn_mask: torch.Tensor, 
                 expected_actions: List[Dict[str, Any]], 
                 attn_kl_weight=0.0):
        attn_kl, contrastive_predictions = self(table_embs_states, no_flight_emb, res_emb, 
                                                token_states, attn_mask)
        logs = {}
        contrastive_loss, constrastive_logs, contrastive_postproc_fs = \
            self.contrastive_table_head.get_loss(contrastive_predictions, 
                                                 expected_actions)
        loss = contrastive_loss
        logs['contrastive'] = constrastive_logs
        loss += attn_kl * attn_kl_weight
        logs['attn_kl'] = (attn_kl.item(), len(expected_actions))
        logs['loss'] = (loss.item(), len(expected_actions))
        return loss, logs, contrastive_postproc_fs




