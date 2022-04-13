import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.base import BaseTransformer, ConstraintParser, ConstraintRewardFunction, Evaluator
from ad.airdialogue import CustomerScenario, Event, Scene
from typing import Any, Dict, Union, List, Optional
from utils.misc import PrecisionRecallAcc, stack_dicts, unstack_dicts
from utils.sampling_utils import *
import numpy as np
from utils.top_k_constraints import top_k_constraints
from utils.torch_utils import get_transformer_logs
from transformers import RobertaModel, RobertaTokenizer
from utils.data_utils import DiscreteFeatures

class ConstraintPredictorRoberta(ConstraintParser, BaseTransformer, nn.Module):
    def __init__(self, 
                 discrete_features: DiscreteFeatures, 
                 roberta_type: str = "roberta-base", 
                 device: Union[torch.device, str] = "cuda", 
                 max_length: Optional[int] = None):
        nn.Module.__init__(self)
        self.roberta_type = roberta_type
        tokenizer = RobertaTokenizer.from_pretrained(self.roberta_type)
        model = RobertaModel.from_pretrained(self.roberta_type)
        BaseTransformer.__init__(self, model, tokenizer, device, max_length)
        ConstraintParser.__init__(self)
        self.h_dim = self.model.config.hidden_size
        self.discrete_features = discrete_features
        self.ffs = nn.ModuleDict({k: nn.Linear(self.h_dim, self.h_dim*2)
                                  for k, _ in self.discrete_features.get_emb_spec().items()})
        self.prediction_heads = nn.ModuleDict({k: nn.Linear(self.h_dim*2, num_items)
                                               for k, num_items in self.discrete_features.get_emb_spec().items()})
        self.attn_proj = nn.Linear(self.h_dim, 1)
        self.param_groups = [
                             ((self.attn_proj, self.ffs, self.prediction_heads,), lambda config: {'lr': config['lr']}), 
                             ((self.model,), lambda config: {'lr': config['roberta_lr'],}), 
                            ]
    
    def format(self, dialogue_events: List[Event], scene: Scene):
        return ("<s> " + self.format_events(dialogue_events)).strip()
    
    def _tokenize_customer_scenarios_batch(self, scenarios: List[CustomerScenario]):
        return {k: torch.tensor(v).to(self.device) for k, v in stack_dicts([scenario.get_discrete_state(self.discrete_features) 
                                                                            for scenario in scenarios]).items()}
    
    def forward(self, tokens: torch.Tensor, attn_mask: torch.Tensor, **kwargs):
        model_output = self.model(input_ids=tokens, attention_mask=attn_mask, **kwargs)
        hidden_states = model_output.last_hidden_state
        hidden_attn = F.softmax(self.attn_proj(hidden_states).squeeze(2).masked_fill_(attn_mask==0, float('-inf')) / math.sqrt(self.h_dim), dim=1)
        constraint_emb = torch.einsum('btd,bt->bd', hidden_states, hidden_attn)
        prediction_logits = {}
        for k, head in self.prediction_heads.items():
            prediction_logits[k] = head(F.relu(self.ffs[k](constraint_emb)))
        return prediction_logits, model_output
    
    def get_loss(self, scenes: List[Scene]):
        tokens, attn_mask = self._tokenize_batch([scene.events for scene in scenes], scenes)
        prediction_logits, model_outputs = self(tokens, attn_mask, output_attentions=True)
        transformer_logs = get_transformer_logs(model_outputs.attentions, self.model, attn_mask)

        labels = self._tokenize_customer_scenarios_batch([scene.customer_scenario for scene in scenes])

        losses = {}
        accuracies = {}
        loss = 0.0
        n = len(scenes)
        exact_match_accuracy = torch.ones((n,)).to(self.device)
        for k in prediction_logits.keys():
            losses[k] = F.cross_entropy(prediction_logits[k], labels[k])
            accuracies[k] = (labels[k] == torch.argmax(prediction_logits[k], dim=-1)).float().mean()
            exact_match_accuracy *= (labels[k] == torch.argmax(prediction_logits[k], dim=-1)).float()
            loss += losses[k]
        logs = {**{k+'_loss': (v.item(), n) for k, v in losses.items()}, **{k+'_acc': (v.item(), n) for k, v in accuracies.items()}}
        logs['exact_match_acc'] = (exact_match_accuracy.mean().item(), n)
        logs['loss'] = (loss.item(), n)
        logs['transformer'] = transformer_logs
        return loss, logs, []
    
    def top_k_constraint_sets(self, scenes: List[Scene], 
                              dialogue_events_list: List[List[Event]], 
                              k: int):
        tokens, attn_mask = self._tokenize_batch(dialogue_events_list, scenes)
        prediction_logits, _ = self(tokens, attn_mask)
        constraint_probs = unstack_dicts({k: F.softmax(v, dim=-1).detach().cpu().tolist() for k, v in prediction_logits.items()})
        constraints = []
        for constraint_prob in constraint_probs:
            constraints.append(list(top_k_constraints(constraint_prob, self.discrete_features.idx2val, k_limit=k)))
        return constraints

class ConstraintPredictorRobertaEvaluator(Evaluator):
    def __init__(self, k=1) -> None:
        super(ConstraintPredictorRobertaEvaluator, self).__init__()
        self.k = k
    
    def evaluate(self, model: ConstraintPredictorRoberta, scenes: List[Scene]) -> Optional[Dict[str, Any]]:
        reward_function = ConstraintRewardFunction(model)
        reward_stats = PrecisionRecallAcc([0, 1])

        predicted_rewards = reward_function.get_reward(scenes, [scene.events for scene in scenes], k=self.k)
        true_rewards = [scene.events[-1].em_reward()['reward'] for scene in scenes]
        assert len(predicted_rewards) == len(true_rewards)
        for i in range(len(true_rewards)):
            reward_stats.add_item(predicted_rewards[i], true_rewards[i], predicted_rewards[i] == true_rewards[i])
        logs = {'reward': reward_stats.return_summary()}
        return logs

