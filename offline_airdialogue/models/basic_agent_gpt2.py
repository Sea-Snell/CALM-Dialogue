import torch
from models.base import GPT2LMBase, Evaluator
from ad.airdialogue import AgentScenario, Event, Scene
from typing import Union, List, Optional, Dict, Any
from models.contrastive_table_head import TransformerTableHead
from utils.data_utils import DiscreteFeatures
from utils.misc import stack_dicts
from utils.sampling_utils import *
from utils.torch_utils import get_transformer_logs

class BasicAgentGPT2(GPT2LMBase):
    def __init__(self, 
                 discrete_features: DiscreteFeatures, 
                 gpt2_type: str = "gpt2", 
                 device: Union[torch.device, str] = "cuda", 
                 max_length: Optional[int] = None, 
                 attn_prior: str = 'geometric', 
                 geometric_rate: float = 0.9):
        super(BasicAgentGPT2, self).__init__(gpt2_type=gpt2_type, 
                                             device=device, 
                                             max_length=max_length)
        self.table_head = TransformerTableHead(discrete_features, self.h_dim, 
                                               device=self.device, attn_prior=attn_prior, 
                                               geometric_rate=geometric_rate)
    
    def format(self, dialogue_events: List[Event], scene: Scene):
        return ("<s> " + self.format_events(dialogue_events)).strip()
    
    def get_loss(self, scenes: List[Scene], 
                 table_head_weight=1.0, 
                 attn_kl_weight=0.0):
        tokens, attn_mask = self._tokenize_batch([scene.events for scene in scenes], scenes)
        prefix_embs = self.table_head.embed_agent_context([scene.agent_scenario for scene in scenes])
        model_outputs = self(prefix_embs, tokens, attn_mask, output_hidden_states=True, output_attentions=True)
        logits = model_outputs.logits
        hidden_state = model_outputs.hidden_states[-1]
        expected_actions = [scene.expected_action for scene in scenes]

        logs = {}
        # print(attn_mask.shape, hidden_state[:, prefix_embs.shape[1]:, :].shape, prefix_embs.shape, attn_mask[:, :(self.max_length-prefix_embs.shape[1])].shape)
        table_head_loss, table_head_logs, table_head_postproc_fs = self.table_head.get_loss(hidden_state[:, 1:(prefix_embs.shape[1]-1), :], 
                                                                                            hidden_state[:, 0, :], 
                                                                                            hidden_state[:, prefix_embs.shape[1]-1, :], 
                                                                                            hidden_state[:, prefix_embs.shape[1]:, :], 
                                                                                            attn_mask[:, :(self.max_length-prefix_embs.shape[1])], 
                                                                                            expected_actions, 
                                                                                            attn_kl_weight=attn_kl_weight)
        transformer_logs = get_transformer_logs(model_outputs.attentions, self.model, 
                                                torch.cat((torch.ones((*prefix_embs.shape[:2],)).to(self.device), attn_mask), dim=1)[:, :self.max_length])
        predictions = logits[:, prefix_embs.shape[1]:-1, :].reshape(-1, logits.shape[-1])
        labels = tokens[:, 1:(self.max_length-prefix_embs.shape[1])].reshape(-1)
        loss = F.cross_entropy(predictions, labels,
                               ignore_index=self.tokenizer.pad_token_id)
        n = (labels != self.tokenizer.pad_token_id).float().sum()
        logs['token_loss'] = (loss.item(), n)
        loss += table_head_weight * table_head_loss
        logs['table_head'] = table_head_logs
        logs['transformer'] = transformer_logs
        postproc_f = lambda l: l.update({'loss': l['token_loss'] + table_head_weight * l['table_head']['loss']})
        return loss, logs, table_head_postproc_fs+[postproc_f]
    
    def fetch_inputs_infer(self, scenes: List[Scene], dialogue_events_list: List[List[Event]]):
        tokens, attn_mask = self._tokenize_batch(dialogue_events_list, scenes)
        prefix_embs = self.table_head.embed_agent_context([scene.agent_scenario for scene in scenes])
        return prefix_embs, tokens, attn_mask
    
    

class BasicAgentGPT2Evaluator(Evaluator):
    def __init__(self, max_generation_len=None, 
                 temp=1.0, top_k=None, top_p=None,
                 verbose=True) -> None:
        super(BasicAgentGPT2Evaluator, self).__init__()
        self.max_generation_len = max_generation_len
        self.temp = temp
        self.top_k = top_k
        self.top_p = top_p
        self.verbose = verbose
    
    def evaluate(self, model: BasicAgentGPT2, scenes: List[Scene]) -> Optional[Dict[str, Any]]:
        if self.verbose:
            predictions, _ = model.sample_full(scenes, [[]]*len(scenes), 
                                               max_generation_len=self.max_generation_len, 
                                               temp=self.temp, 
                                               top_k=self.top_k, 
                                               top_p=self.top_p)
            for prefix, prediction in predictions:
                prediction = prediction[0]
                print('======================')
                print(prefix)
                print(prediction)
                print('======================')
                print()
                print()
        return None

    

