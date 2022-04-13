import torch
from bots.base_bot import BaseBot
from models.aux_head2 import Aux2
from models.base import GPT2LMBase, Evaluator, GPT2LMPolicy
from ad.airdialogue import AgentScenario, Event, Scene
from typing import Union, List, Optional, Dict, Any
from models.contrastive_table_head import TransformerTableHead
from models.embedding_modules import EmbeddingCombiner
from utils.data_utils import DiscreteFeatures
from utils.misc import PrecisionRecallAcc, stack_dicts
from utils.sampling_utils import *
from utils.torch_utils import get_transformer_logs
from bots.basic_policy_bot import BasicPolicyBot
from selfplay import selfplay

class SimplifiedAuxGPT2(GPT2LMBase):
    def __init__(self, 
                 discrete_features: DiscreteFeatures, 
                 gpt2_type: str = "gpt2", 
                 device: Union[torch.device, str] = "cuda", 
                 max_length: Optional[int] = None):
        super(SimplifiedAuxGPT2, self).__init__(gpt2_type=gpt2_type, 
                                             device=device, 
                                             max_length=max_length)
        self.discrete_features = discrete_features
        self.aux_head = Aux2(self.h_dim, self.device)
        self.flight_encoder = EmbeddingCombiner(self.discrete_features.get_emb_spec(), 
                                                self.h_dim)
    
    def format(self, dialogue_events: List[Event], scene: Scene):
        return ("<s> " + self.format_events(dialogue_events)).strip()
    
    def _tokenize_agent_scenarios_batch(self, scenarios: List[AgentScenario]):
        return {k: torch.tensor(v).to(self.device) for k, v in stack_dicts([scenario.get_discrete_state(self.discrete_features) 
                                                                            for scenario in scenarios]).items()}
    
    def embed_agent_context(self, agent_states: List[AgentScenario]):
        flight_tables = self._tokenize_agent_scenarios_batch(agent_states)
        embeddings = self.flight_encoder(flight_tables)
        return embeddings

    def get_loss(self, scenes: List[Scene], 
                 aux_head_weight=1.0):
        tokens, attn_mask = self._tokenize_batch([scene.events for scene in scenes], scenes)
        prefix_embs = self.embed_agent_context([scene.agent_scenario for scene in scenes])
        model_outputs = self(prefix_embs, tokens, attn_mask, output_hidden_states=True, output_attentions=True)
        logits = model_outputs.logits
        hidden_state = model_outputs.hidden_states[-1]
        # expected_actions = [scene.expected_action for scene in scenes]
        expected_actions = [scene.events[-1].event.to_json() for scene in scenes]

        logs = {}
        aux_head_loss, aux_head_logs, aux_head_postproc_fs = self.aux_head.get_loss(hidden_state[:, prefix_embs.shape[1]:, :], 
                                                                                    attn_mask, expected_actions)
        transformer_logs = get_transformer_logs(model_outputs.attentions, self.model, 
                                                torch.cat((torch.ones((*prefix_embs.shape[:2],)).to(self.device), attn_mask), dim=1)[:, :self.max_length])
        predictions = logits[:, prefix_embs.shape[1]:-1, :].reshape(-1, logits.shape[-1])
        labels = tokens[:, 1:(self.max_length-prefix_embs.shape[1])].reshape(-1)
        loss = F.cross_entropy(predictions, labels,
                               ignore_index=self.tokenizer.pad_token_id)
        n = (labels != self.tokenizer.pad_token_id).float().sum()
        logs['token_loss'] = (loss.item(), n)
        loss += aux_head_weight * aux_head_loss
        logs['aux_head'] = aux_head_logs
        logs['transformer'] = transformer_logs
        postproc_f = lambda l: l.update({'loss': l['token_loss'] + aux_head_weight * l['aux_head']['loss']})
        return loss, logs, aux_head_postproc_fs+[postproc_f]
    
    def fetch_inputs_infer(self, scenes: List[Scene], dialogue_events_list: List[List[Event]]):
        tokens, attn_mask = self._tokenize_batch(dialogue_events_list, scenes)
        prefix_embs = self.embed_agent_context([scene.agent_scenario for scene in scenes])
        return prefix_embs, tokens, attn_mask
    
    
class SimplifiedAuxGPT2Evaluator(Evaluator):
    def __init__(self, 
                 opposing_bot: BaseBot, 
                 max_turns=80, 
                 verbose=True, 
                 kind='sample', 
                 **generation_kwargs) -> None:
        super(SimplifiedAuxGPT2Evaluator, self).__init__()
        self.opposing_bot = opposing_bot
        self.opposing_bot.eval()
        self.max_turns = max_turns
        self.verbose = verbose
        self.kind = kind
        self.generation_kwargs = generation_kwargs
        self.expected_f = None
        if self.kind == 'sample':
            self.expected_f = 'sample_one'
        elif self.kind == 'beam':
            self.expected_f = 'beam_one'
        else:
            raise NotImplementedError
    
    def evaluate(self, model: SimplifiedAuxGPT2, scenes: List[Scene]) -> Optional[Dict[str, Any]]:
        policy = GPT2LMPolicy(model, self.kind)
        self_bot = BasicPolicyBot(policy, **self.generation_kwargs)
        self_bot.eval()
        reward_stats = PrecisionRecallAcc([0, 1])
        for scene in scenes:
            selfplay_reward, _ = selfplay(self.opposing_bot, self_bot, scene, max_turns=self.max_turns, verbose=self.verbose)
            predicted_reward = selfplay_reward['reward']
            actual_reward = scene.events[-1].em_reward()['reward']
            reward_stats.add_item(predicted_reward, actual_reward, predicted_reward == actual_reward)
        logs = {'reward': reward_stats.return_summary()}
        return logs

    

