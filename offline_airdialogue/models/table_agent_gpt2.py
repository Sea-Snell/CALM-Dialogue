import torch
from bots.base_bot import BaseBot
from models.base import GPT2LMBase, Evaluator, GPT2LMPolicy
from models.contrastive_table_head import TransformerTableHead
from bots.basic_policy_bot import BasicPolicyBot
from selfplay import selfplay
from ad.airdialogue import Event, Scene
from typing import Union, List, Optional, Dict, Any
from utils.data_utils import DiscreteFeatures
from utils.sampling_utils import *
from utils.misc import PrecisionRecallAcc, strip_from_beginning, strip_from_end
from utils.torch_utils import get_transformer_logs

class TableGPT2Agent(GPT2LMBase):
    def __init__(self, 
                 discrete_features: DiscreteFeatures, 
                 cond_reward_key: str, 
                 gpt2_type: str = "gpt2", 
                 device: Union[torch.device, str] = "cuda", 
                 max_length: Optional[int] = None, 
                 attn_prior: str = 'geometric', 
                 geometric_rate: float = 0.9):
        super(TableGPT2Agent, self).__init__(gpt2_type=gpt2_type, 
                                             device=device, 
                                             max_length=max_length)
        self.cond_reward_key = cond_reward_key
        self.table_head = TransformerTableHead(discrete_features, self.h_dim, 
                                               device=self.device, attn_prior=attn_prior, 
                                               geometric_rate=geometric_rate)

    def format(self, scene: Scene, dialogue_events: List[Event], add_expected_action=False):
        reward = scene.data[self.cond_reward_key]
        if self.cond_reward_key not in scene.data:
            reward = dialogue_events[-1].em_reward()['reward']
        else:
            reward = scene.data[self.cond_reward_key]
        dialogue_str = f'<s> {self.format_reward(reward)} {self.format_events(dialogue_events)}'
        if add_expected_action:
            dialogue_str += f' {self.format_expected_action(scene.expected_action)}'
        dialogue_str = dialogue_str.strip()
        return dialogue_str
    
    def format_expected_action(self, expected_action: Dict[str, Any]):
        expected_action = dict(expected_action)
        order = ["status", "name", "flight"]
        if len(expected_action["flight"]) == 0:
            expected_action["flight"] = "0"
        else:
            expected_action["flight"] = " , ".join(map(str, expected_action["flight"]))
        return f'Expected: {" , ".join([expected_action[item] for item in order])} </s>'
    
    def parse_expected_action(self, expected_str, raise_error=False):
        expected_str = expected_str.strip()
        expected_str = strip_from_end(expected_str, '</s>').strip()
        expected_str = strip_from_beginning(expected_str, '<s>').strip()
        expected_str = strip_from_beginning(expected_str, 'Expected:').strip()
        expected_items = list(map(lambda x: x.strip(), expected_str.split(",")))
        try:
            action = {'status': expected_items[0], 'name': expected_items[1],
                      'flight': list(map(int, expected_items[2:]))}
            if action['flight'] == [0]:
                action['flight'] = []
            return action
        except Exception as e:
            if raise_error:
                raise e
            return None
    
    def format_reward(self, reward: float):
        return f"{int(reward)} </s>"
    
    def get_loss(self, 
                 scenes: List[Scene], 
                 table_head_weight=1.0, 
                 attn_kl_weight=0.0):
        tokens, attn_mask = self._tokenize_batch(scenes, [scene.events for scene in scenes], 
                                                 [True]*len(scenes))
        prefix_embs = self.table_head.embed_agent_context([scene.agent_scenario for scene in scenes])
        model_outputs = self(prefix_embs, tokens, attn_mask, output_hidden_states=True, output_attentions=True)
        logits = model_outputs.logits
        hidden_state = model_outputs.hidden_states[-1]
        expected_actions = [scene.expected_action for scene in scenes]
        
        logs = {}
        table_head_loss, table_head_logs, table_head_postproc_fs = self.table_head.get_loss(hidden_state[:, 1:(prefix_embs.shape[1]-1), :], 
                                                                                            hidden_state[:, 0, :], 
                                                                                            hidden_state[:, prefix_embs.shape[1]-1, :], 
                                                                                            hidden_state[:, prefix_embs.shape[1]:, :], 
                                                                                            attn_mask, 
                                                                                            expected_actions, 
                                                                                            attn_kl_weight=attn_kl_weight)
        transformer_logs = get_transformer_logs(model_outputs.attentions, self.model, 
                                                torch.cat((torch.ones((*prefix_embs.shape[:2],)).to(self.device), attn_mask), dim=1))
        predictions = logits[:, prefix_embs.shape[1]:-1, :].reshape(-1, logits.shape[-1])
        labels = tokens[:, 1:].reshape(-1)
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
        tokens, attn_mask = self._tokenize_batch(scenes, dialogue_events_list, [False]*len(scenes))
        prefix_embs = self.table_head.embed_agent_context([scene.agent_scenario for scene in scenes])
        return prefix_embs, tokens, attn_mask

class TableGPT2AgentEvaluator(Evaluator):
    def __init__(self, 
                 opposing_bot: BaseBot, 
                 max_turns=80, 
                 verbose=True, 
                 kind='sample', 
                 **generation_kwargs) -> None:
        super(TableGPT2AgentEvaluator, self).__init__()
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
    
    def evaluate(self, model: TableGPT2Agent, scenes: List[Scene]) -> Optional[Dict[str, Any]]:
        policy = GPT2LMPolicy(model, self.kind)
        self_bot = BasicPolicyBot(policy, **self.generation_kwargs)
        expected_action_stats = PrecisionRecallAcc(['no_flight', 'cancel', 'book', 'change', 'no_reservation'])
        reward_stats = PrecisionRecallAcc([0, 1])
        self_reward_stats = PrecisionRecallAcc([0, 1])
        for scene in scenes:
            selfplay_reward, final_event = selfplay(self.opposing_bot, self_bot, scene, max_turns=self.max_turns, verbose=self.verbose)
            generations, probs = getattr(model, self.expected_f)([scene], [final_event.get_events()], 
                                                                 **self.generation_kwargs)
            expected = select_max_prob_response(generations[0], probs[0].detach().cpu().numpy())
            if self.verbose:
                print('predicted expected actions:', expected)
                print('==============================')
                print()
                print()

            predicted_reward = selfplay_reward['reward']
            actual_reward = scene.events[-1].em_reward()['reward']
            reward_stats.add_item(predicted_reward, actual_reward, predicted_reward == actual_reward)

            predicted_expected_action = model.parse_expected_action(expected, raise_error=False)
            if predicted_expected_action is None:
                predicted_expected_action = {'status': 'ERROR', 'flight': [], 'name': ''}
            self_reward = final_event.em_reward(expected_action=predicted_expected_action)['reward']
            self_reward_stats.add_item(self_reward, predicted_reward, self_reward == predicted_reward)

            predicted_expected_action['flight'] = set(predicted_expected_action['flight'])
            true_expected_action = scene.expected_action.copy()
            true_expected_action['flight'] = set(true_expected_action['flight'])
            expected_action_stats.add_item(predicted_expected_action['status'], 
                                           true_expected_action['status'], 
                                           true_expected_action == predicted_expected_action)
        logs = {'expected_action': expected_action_stats.return_summary(), 'reward': reward_stats.return_summary(), 
                'self_reward': self_reward_stats.return_summary()}
        return logs


