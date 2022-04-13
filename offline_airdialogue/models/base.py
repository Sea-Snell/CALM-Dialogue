import math
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from ad.airdialogue import Event, Scene
from ad.query_table import get_true_action
from utils.sampling_utils import end_of_dialogue_termination_condition, single_utterance_termination_condition
from typing import Dict, List, Optional, Callable, Any, Union
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel
import numpy as np
import functools
from utils.sampling_utils import *
from utils.torch_utils import to
import requests
import json
import base64
from airdialogue.evaluator.metrics import bleu

class RewardFunction(ABC):
    @abstractmethod
    def get_reward(self, scenes: List[Scene], 
                   dialogue_events_list: List[List[Event]], 
                   **kwargs) -> np.ndarray:
        pass
    
    def eval(self):
        pass

    def train(self):
        pass

class Policy(ABC):
    @abstractmethod
    def generate_one(self, scenes: List[Scene], 
                     dialogue_events_list: List[List[Event]], 
                     n: int = 1, 
                     **kwargs) -> Tuple[List[List[Event]], np.ndarray]:
        pass

    @abstractmethod
    def generate_full(self, scenes: List[Scene], 
                      dialogue_events_list: List[List[Event]], 
                      n: int = 1, 
                      **kwargs) -> Tuple[List[List[Event]], np.ndarray]:
        pass
    
    def eval(self):
        pass

    def train(self):
        pass

class ConstraintParser(ABC):
    @abstractmethod
    def top_k_constraint_sets(self, scenes: List[Scene], 
                              dialogue_events_list: List[List[Event]], 
                              k: int = 1):
        pass

class BaseTransformer(ABC, nn.Module):
    def __init__(self, 
                 pretrained_model: PreTrainedModel, 
                 tokenizer: PreTrainedTokenizerBase, 
                 device: Union[torch.device, str] = "cuda", 
                 max_length: Optional[int] = None) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.tokenizer.add_special_tokens({'additional_special_tokens': list(map(str, range(1000, 1030))),
                                           'bos_token': '<s>',
                                           'sep_token': '</s>',
                                           'pad_token': '<|pad|>'})
        self.model = pretrained_model
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.max_length = self.tokenizer.model_max_length if max_length is None else max_length
        self.max_length = min(self.max_length, self.tokenizer.model_max_length)
        self.device = device

    @abstractmethod
    def format(self, *args):
        pass

    def format_events(self, dialogue_events: List[Event]):
        dialog_p: List[str] = []
        for d in dialogue_events:
            dialog_p.append(f'{d.get_speaker()}: {str(d.event)}')
        dialogue_str = " </s> ".join(dialog_p)
        if len(dialogue_events) > 0:
            dialogue_str += " </s>"
        return dialogue_str

    def _encode(self, sent, **kwargs):
        # might want to check that add_special_tokens=False doesn't mess up gpt2, it works with bart
        return to(dict(self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
            **kwargs,
        )), self.device)

    def _decode(self, sent, **kwargs):
        return self.tokenizer.decode(sent.tolist(), **kwargs)

    def _tokenize_batch_strs(self, strs):
        encoded_items = self._encode(strs, padding=True)
        tokens, attn_mask = encoded_items['input_ids'], encoded_items['attention_mask']
        return tokens, attn_mask

    def _tokenize_batch(self, *args, formatter='format'):
        formatter = getattr(self, formatter)
        return self._tokenize_batch_strs(list(map(lambda x: formatter(*x), zip(*args))))

    @abstractmethod
    def get_loss(self, scenes: List[Scene], **kwargs):
        pass

class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, model: BaseTransformer, scenes: List[Scene]) -> Optional[Dict[str, Any]]:
        pass

    def postproc(self, logs: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        return logs

class GPT2LMBase(BaseTransformer, nn.Module):
    def __init__(self, 
                 gpt2_type: str = "gpt2", 
                 device: Union[torch.device, str] = "cuda", 
                 max_length: Optional[int] = None) -> None:
        nn.Module.__init__(self)
        self.gpt2_type = gpt2_type
        tokenizer = GPT2Tokenizer.from_pretrained(self.gpt2_type)
        model = GPT2LMHeadModel.from_pretrained(self.gpt2_type)
        BaseTransformer.__init__(self, model, tokenizer, device, max_length)
        self.h_dim = self.model.config.n_embd

    def forward(self, prefix_embs: Optional[torch.Tensor], tokens: torch.Tensor, attn_mask: torch.Tensor, **kwargs):
        # prefix_embs – b,t,d
        # tokens – b,t
        # attn_mask – b,t
        # assumes prefix dims is the same in time dim for all batch items (i.e. no padding)
        if prefix_embs is None:
            prefix_embs = torch.empty((tokens.shape[0], 0, self.h_dim)).to(self.device)
        input_embeddings = torch.cat((prefix_embs, self.model.transformer.wte(tokens)), dim=1)[:, :self.max_length, :]
        input_attn_mask = torch.cat((torch.ones(prefix_embs.shape[:2]).to(self.device), attn_mask), dim=1)[:, :self.max_length]
        model_outputs = self.model(inputs_embeds=input_embeddings, attention_mask=input_attn_mask, **kwargs)
        return model_outputs

    @abstractmethod
    def fetch_inputs_infer(self, scenes: List[Scene], dialogue_events_list: List[List[Event]]):
        pass

    # def sample_raw_batch(self, 
    #                      prefix_embs: Optional[torch.Tensor], 
    #                      tokens: torch.Tensor, attn_mask: torch.Tensor, 
    #                      batch_size: int, 
    #                      termination_condition: Callable[[np.ndarray], bool], 
    #                      num_generations=1, max_generation_len=None, 
    #                      temp=1.0, top_k=None, top_p=None):
    #     pass
    
    def sample_raw(self, 
                   prefix_embs: Optional[torch.Tensor], 
                   tokens: torch.Tensor, attn_mask: torch.Tensor, 
                   termination_condition: Callable[[np.ndarray], bool], 
                   num_generations=1, max_generation_len=None, 
                   temp=1.0, top_k=None, top_p=None):
        bsize = tokens.shape[0]
        n = bsize * num_generations
        if max_generation_len is None:
            max_generation_len = self.max_length+1
        if prefix_embs is None:
            prefix_embs = torch.empty((tokens.shape[0], 0, self.h_dim)).to(self.device)
        prefix_t = prefix_embs.shape[1]
        dialogue_strs = [self._decode(tokens[i, :][:attn_mask[i, :].sum()], clean_up_tokenization_spaces=False) for i in range(len(tokens))]
        model_outputs = self(prefix_embs, tokens, attn_mask, use_cache=True)
        dialogue_kvs = model_outputs.past_key_values
        dialogue_lens = attn_mask.sum(dim=1)

        tokens = pad_sequence(torch.repeat_interleave(tokens, num_generations, dim=0), self.max_length, self.tokenizer.pad_token_id, self.device, 1)
        dialogue_lens = torch.repeat_interleave(dialogue_lens, num_generations, dim=0)
        dialogue_kvs = map_all_kvs(lambda x: pad_sequence(torch.repeat_interleave(x, num_generations, dim=0), self.max_length, 0.0, self.device, 2), dialogue_kvs)
        log_probs = torch.full((dialogue_lens.shape[0],), 0.0).to(self.device)
        termination_mask = torch.full((dialogue_lens.shape[0],), 1).to(self.device)
        t = torch.min(dialogue_lens).int()
        while termination_mask.sum() > 0 and (t+prefix_t) < self.max_length:
            curr_token = tokens[:, t-1].unsqueeze(1)
            curr_dialogue_kvs = map_all_kvs(lambda x: x[:,:,:(t+prefix_t)-1,:], dialogue_kvs)
            transformer_outputs = self.model(curr_token, past_key_values=curr_dialogue_kvs, use_cache=True)
            logits = transformer_outputs.logits
            logits[:, 0, self.tokenizer.pad_token_id] = torch.where(termination_mask == 1, float('-inf'), 1e7)
            logits[torch.arange(0, n).to(self.device), torch.full((n,), 0).to(self.device), tokens[:, t]] = logits[torch.arange(0, n).to(self.device), torch.full((n,), 0).to(self.device), tokens[:, t]].masked_fill_(t < dialogue_lens, 1e7)
            logits = process_logits(transformer_outputs.logits, temp=temp, top_k=top_k, top_p=top_p)
            cat_dist = torch.distributions.categorical.Categorical(logits=logits[:, 0])
            new_tokens = cat_dist.sample()
            log_probs += cat_dist.log_prob(new_tokens)
            tokens[:, t] = new_tokens
            dialogue_kvs = update_kvs(dialogue_kvs, transformer_outputs.past_key_values, torch.arange(0, n).to(self.device), (t+prefix_t)-1)
            for idx in range(n):
                if tokens[idx, t] == self.tokenizer.sep_token_id and t >= dialogue_lens[idx]:
                    termination_mask[idx] *= (1 - int(termination_condition(self._decode(tokens[idx, :],
                                                                                         clean_up_tokenization_spaces=False))))
            termination_mask *= ((t-dialogue_lens) < max_generation_len).int()
            t += 1
    
        output_strs = [self._decode(tokens[i, :], clean_up_tokenization_spaces=False) for i in range(len(tokens))]
        processed_outputs = []
        for i in range(len(dialogue_strs)):
            temp_outputs = []
            for x in range(num_generations):
                processed_str = output_strs[i*num_generations+x][len(dialogue_strs[i]):].strip()
                if '<|pad|>' in processed_str:
                    processed_str = processed_str[:processed_str.find('<|pad|>')].strip()
                temp_outputs.append(processed_str)
            processed_outputs.append(temp_outputs)
        return list(zip(dialogue_strs, processed_outputs)), log_probs.reshape(-1, num_generations)
    
    def beam_raw(self, 
                 prefix_embs: Optional[torch.Tensor], 
                 tokens: torch.Tensor, attn_mask: torch.Tensor, 
                 termination_condition: Callable[[np.ndarray], bool], 
                 beam_width=1, max_generation_len=None):
        bsize, vocab_size = tokens.shape[0], len(self.tokenizer)
        n = bsize * beam_width
        if max_generation_len is None:
            max_generation_len = self.max_length+1
        if prefix_embs is None:
            prefix_embs = torch.empty((tokens.shape[0], 0, self.h_dim)).to(self.device)
        prefix_t = prefix_embs.shape[1]
        dialogue_strs = [self._decode(tokens[i, :][:attn_mask[i, :].sum()], clean_up_tokenization_spaces=False) for i in range(len(tokens))]
        model_outputs = self(prefix_embs, tokens, attn_mask, use_cache=True)
        dialogue_kvs = model_outputs.past_key_values
        original_dialogue_lens = attn_mask.sum(dim=1)
        batch_indicator = torch.stack(beam_width*[torch.arange(0, bsize).to(self.device)], dim=1)

        tokens = pad_sequence(torch.repeat_interleave(tokens, beam_width, dim=0), self.max_length, self.tokenizer.pad_token_id, self.device, 1)
        dialogue_lens = torch.repeat_interleave(original_dialogue_lens, beam_width, dim=0)
        dialogue_kvs = map_all_kvs(lambda x: pad_sequence(torch.repeat_interleave(x, beam_width, dim=0), self.max_length, 0.0, self.device, 2), dialogue_kvs)
        curr_scores = torch.zeros(bsize, beam_width).to(self.device)  # (batch, k)
        termination_mask = torch.full((n,), 1).to(self.device)
        t = torch.min(dialogue_lens).int()
        while termination_mask.sum() > 0 and (t+prefix_t) < self.max_length:
            curr_token = tokens[:, t-1].unsqueeze(1)
            curr_dialogue_kvs = map_all_kvs(lambda x: x[:,:,:(t+prefix_t)-1,:], dialogue_kvs)
            transformer_outputs = self.model(curr_token, past_key_values=curr_dialogue_kvs, use_cache=True)
            logits = transformer_outputs.logits
            logits[:, 0, self.tokenizer.pad_token_id] = torch.where(termination_mask == 1, float('-inf'), 1e7)
            logits[torch.arange(0, n).to(self.device), torch.full((n,), 0).to(self.device), tokens[:, t]] = logits[torch.arange(0, n).to(self.device), torch.full((n,), 0).to(self.device), tokens[:, t]].masked_fill_(t < dialogue_lens, 1e7)
            scores = (torch.log(F.softmax(logits, dim=-1)).reshape(1, bsize, beam_width, -1).permute(3, 0, 1, 2) + curr_scores).permute(1, 2, 3, 0).reshape(1, bsize, -1)  # (time, batch, k*vocab)
            scores[0, :, vocab_size:] = scores[0, :, vocab_size:].masked_fill_((t == original_dialogue_lens).unsqueeze(1).repeat(1, scores.shape[2]-vocab_size), float('-inf'))
            curr_scores, top_k = torch.topk(scores[0, :, :], k=beam_width, dim=1)  # (batch, k), (batch, k)
            tokens = tokens[(batch_indicator * beam_width + (top_k // vocab_size)).reshape(-1), :]
            tokens[:, t] = top_k.reshape(-1) % vocab_size  # (batch*k,)
            fixed_dialogue_kvs = map_all_kvs(lambda x: x[(batch_indicator * beam_width + (top_k // vocab_size)).reshape(-1), :, :, :], transformer_outputs.past_key_values)
            dialogue_kvs = map_all_kvs(lambda x: x[(batch_indicator * beam_width + (top_k // vocab_size)).reshape(-1), :, :, :], dialogue_kvs)
            dialogue_kvs = update_kvs(dialogue_kvs, fixed_dialogue_kvs, torch.arange(0, n).to(self.device), (t+prefix_t)-1)
            dialogue_lens = dialogue_lens[(batch_indicator * beam_width + (top_k // vocab_size)).reshape(-1)]
            termination_mask = termination_mask[(batch_indicator * beam_width + (top_k // vocab_size)).reshape(-1)]
            for idx in range(n):
                if tokens[idx, t] == self.tokenizer.sep_token_id and t >= dialogue_lens[idx]:
                    termination_mask[idx] *= (1 - int(termination_condition(self._decode(tokens[idx, :],
                                                                                         clean_up_tokenization_spaces=False))))
            termination_mask *= ((t-dialogue_lens) < max_generation_len).int()
            t += 1
        output_strs = [self._decode(tokens[i, :], clean_up_tokenization_spaces=False) for i in range(n)]
        processed_outputs = []
        for i in range(len(dialogue_strs)):
            temp_outputs = []
            for x in range(beam_width):
                processed_str = output_strs[i*beam_width+x][len(dialogue_strs[i]):].strip()
                if '<|pad|>' in processed_str:
                    processed_str = processed_str[:processed_str.find('<|pad|>')].strip()
                temp_outputs.append(processed_str)
            processed_outputs.append(temp_outputs)
        return list(zip(dialogue_strs, processed_outputs)), curr_scores
    
    def generate(self, scenes: List[Scene], 
                 dialogue_events_list: List[List[Event]], 
                 termination_condition: Callable[[np.ndarray], bool], 
                 method: str, **kwargs):
        prefix_embs, tokens, attn_mask = self.fetch_inputs_infer(scenes, dialogue_events_list)
        if method == 'beam':
            method = self.beam_raw
        elif method == 'sample':
            method = self.sample_raw
        else:
            raise NotImplementedError
        return method(prefix_embs, tokens, attn_mask, 
                      termination_condition, **kwargs)

    def sample_one(self, scenes: List[Scene], 
                   dialogue_events_list: List[List[Event]], 
                   n=1, max_generation_len=None, 
                   temp=1.0, top_k=None, top_p=None):
        return self.generate(scenes, dialogue_events_list, 
                             single_utterance_termination_condition, 
                             'sample', 
                             num_generations=n, 
                             max_generation_len=max_generation_len, 
                             temp=temp, top_k=top_k, top_p=top_p)

    def sample_full(self, scenes: List[Scene], 
                    dialogue_events_list: List[List[Event]], 
                    n=1, max_generation_len=None, 
                    temp=1.0, top_k=None, top_p=None, final_speaker_str='Submit:'):
        return self.generate(scenes, dialogue_events_list, 
                             functools.partial(end_of_dialogue_termination_condition, 
                                               sep_token=self.tokenizer.sep_token, 
                                               final_speaker_str=final_speaker_str), 
                             'sample', 
                             num_generations=n, 
                             max_generation_len=max_generation_len, 
                             temp=temp, top_k=top_k, top_p=top_p)

    def beam_one(self, scenes: List[Scene], 
                 dialogue_events_list: List[List[Event]], 
                 n=1, max_generation_len=None):
        return self.generate(scenes, dialogue_events_list, 
                             single_utterance_termination_condition, 
                             'beam', 
                             beam_width=n, 
                             max_generation_len=max_generation_len)

    def beam_full(self, scenes: List[Scene], 
                  dialogue_events_list: List[List[Event]], 
                  n=1, max_generation_len=None, 
                  final_speaker_str='Submit:'):
        return self.generate(scenes, dialogue_events_list, 
                             functools.partial(end_of_dialogue_termination_condition, 
                                               sep_token=self.tokenizer.sep_token, 
                                               final_speaker_str=final_speaker_str), 
                             'beam', 
                             beam_width=n, 
                             max_generation_len=max_generation_len)

class GPT2LMPolicy(Policy):
    def __init__(self, lm: GPT2LMBase, kind: str) -> None:
        super().__init__()
        self.lm = lm
        self.kind = kind

    def generate_one(self, scenes: List[Scene], 
                     dialogue_events_list: List[List[Event]], 
                     **generation_kwargs):
        if self.kind == 'beam':
            method = self.lm.beam_one
        elif self.kind == 'sample':
            method = self.lm.sample_one
        else:
            raise NotImplementedError
        generations, probs = method(scenes, 
                                    dialogue_events_list, 
                                    **generation_kwargs)
        outputs = []
        for i, (_, b) in enumerate(generations):
            temp_outputs = []
            for generation in b:
                new_event = parse_utterance(generation, scenes[i], 
                                            dialogue_events_list[i][-1] if len(dialogue_events_list[i]) != 0 else None)
                temp_outputs.append(new_event)
            temp_outputs = list(zip(*sorted(zip(temp_outputs, probs[i]), key=lambda x: -x[1])))[0]
            outputs.append(temp_outputs)
        return outputs, torch.sort(probs, dim=1, descending=True).values.detach().cpu().numpy()

    def generate_full(self, scenes: List[Scene], 
                      dialogue_events_list: List[List[Event]], 
                      **generation_kwargs):
        if self.kind == 'beam':
            method = self.lm.beam_full
        elif self.kind == 'sample':
            method = self.lm.sample_full
        else:
            raise NotImplementedError
        generations, probs = method(scenes, 
                                    dialogue_events_list, 
                                    **generation_kwargs)
        outputs = []
        for i, (_, b) in enumerate(generations):
            temp_outputs = []
            for generation in b:
                new_event = parse_utterances(generation, scenes[i], 
                                             dialogue_events_list[i][-1] if len(dialogue_events_list[i]) != 0 else None)
                temp_outputs.append(new_event)
            temp_outputs = list(zip(*sorted(zip(temp_outputs, probs[i]), key=lambda x: -x[1])))[0]
            outputs.append(temp_outputs)
        return outputs, torch.sort(probs, dim=1, descending=True).values.detach().cpu().numpy()
    
    def eval(self):
        self.lm.eval()

    def train(self):
        self.lm.train()

class RemotePolicy(Policy):
    def __init__(self, url: str):
        super(RemotePolicy, self).__init__()
        self.url = url
        self.headers = {'User-Agent': 'Mozilla/5.0'}
    
    def generate_one(self, scenes: List[Scene], 
                     dialogue_events_list: List[List[Event]], 
                     **generation_kwargs):
        new_events = []
        for i in range(len(scenes)):
            history = list(map(lambda x: (x.get_speaker(), str(x.event)), dialogue_events_list[i]))
            payload = {'history': base64.b64encode(json.dumps(history).encode('utf-8')), 'scenario_idx': scenes[i].scene_id}
            session = requests.Session()
            utterances, terminate, final_action = json.loads(session.post(self.url,headers=self.headers, data=payload).text)
            # might want to account for the final utterance too
            if terminate:
                speaker, response = 'Submit', ', '.join(map(str, final_action))
            else:
                speaker, response = utterances[-1]
            new_event = dialogue_events_list[i][-1].append(Event.from_sentence(speaker, response, scenes[i]))
            new_events.append([new_event])
        return new_events, np.zeros((len(scenes),1))

    def generate_full(self, scenes: List[Scene], 
                      dialogue_events_list: List[List[Event]], 
                      **generation_kwargs):
        raise NotImplementedError

class OracleRewardFunction(RewardFunction):
    def get_reward(self, scenes: List[Scene], 
                   dialogue_events_list: List[List[Event]], 
                   **kwargs):
        return np.array([dialogue_events[-1].em_reward()['reward'] for dialogue_events in dialogue_events_list])

class OracleConstraintParser(ConstraintParser):
    def top_k_constraint_sets(self, scenes: List[Scene], 
                              dialogue_events_list: List[List[Event]], 
                              k: int = 1):
        return [[(scene.customer_scenario.intention, 1.0)] for scene in scenes]

class ConstraintRewardFunction(RewardFunction):
    def __init__(self, constraint_parser: ConstraintParser) -> None:
        super().__init__()
        self.constraint_parser = constraint_parser
    
    def get_reward(self, scenes: List[Scene], 
                   dialogue_events_list: List[List[Event]], 
                   k=1):
        constraints_list = self.constraint_parser.top_k_constraint_sets(scenes, dialogue_events_list, k)
        rewards = []
        for i, constraints in enumerate(constraints_list):
            total_reward = 0.0
            total_prob = 0.0
            for constraint, prob in constraints:
                constraint['name'] = ''
                predicted_true_action = get_true_action(scenes[i].agent_scenario.kb['kb'], 
                                                        scenes[i].agent_scenario.kb['reservation'], 
                                                        constraint)
                total_reward += dialogue_events_list[i][-1].em_reward(expected_action=predicted_true_action, check_name=False)['reward'] * prob
                total_prob += prob
            rewards.append(total_reward / total_prob)
            # rewards.append(total_reward)
        return np.array(rewards)

class LanguageEvaluator(Evaluator):
    def __init__(self, trun_first=False, **generation_kwargs) -> None:
        super(LanguageEvaluator, self).__init__()
        # NOTE: won't work for Multi Process Eval.
        self.trunc_first = trun_first
        self.generation_kwargs = generation_kwargs
        self.all_translations = []
        self.all_references = []
        self.customer_translations = []
        self.customer_references = []
        self.agent_translations = []
        self.agent_references = []
    
    def get_prediction(self, policy: Policy, scenes: List[Scene], dialogue_events_list: List[List[Event]]):
        generations, _ = policy.generate_one(scenes, dialogue_events_list, **self.generation_kwargs)
        return [generation[0] for generation in generations]
    
    def evaluate(self, policy: GPT2LMBase, scenes: List[Scene]) -> Optional[Dict[str, Any]]:
        batch_scenes = []
        batch_events = []
        batch_refs = []
        for scene in scenes:
            for event in (scene.events[1:-1] if self.trunc_first else scene.events[:-1]):
                curr_events = event.get_events()[:-1]
                batch_scenes.append(scene)
                batch_events.append(curr_events)
                batch_refs.append(event)
        bsize = len(scenes)
        for i in range(0, len(batch_scenes), bsize):
            curr_scenes = batch_scenes[i:(i+bsize)]
            curr_events = batch_events[i:(i+bsize)]
            curr_refs = batch_refs[i:(i+bsize)]
            predictions = self.get_prediction(policy, curr_scenes, curr_events)
            self.all_translations.extend([str(prediction.event).split() for prediction in predictions])
            self.all_references.extend([[str(ref.event).split()] for ref in curr_refs])
            self.customer_translations.extend([str(prediction.event).split() for x, prediction in enumerate(predictions) if curr_refs[x].agent == Agent.CUSTOMER])
            self.customer_references.extend([[str(ref.event).split()] for ref in curr_refs if ref.agent == Agent.CUSTOMER])
            self.agent_translations.extend([str(prediction.event).split() for x, prediction in enumerate(predictions) if curr_refs[x].agent == Agent.AGENT])
            self.agent_references.extend([[str(ref.event).split()] for ref in curr_refs if ref.agent == Agent.AGENT])
        return {}
    
    def postproc(self, logs: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        bleu_score_all, _, _, _, _, _ = bleu.compute_bleu(self.all_references, self.all_translations,
                                                          max_order=4, smooth=False)
        bleu_score_customer, _, _, _, _, _ = bleu.compute_bleu(self.customer_references, self.customer_translations,
                                                               max_order=4, smooth=False)
        bleu_score_agent, _, _, _, _, _ = bleu.compute_bleu(self.agent_references, self.agent_translations,
                                                            max_order=4, smooth=False)
        logs['bleu_score_all'] = bleu_score_all*100
        logs['bleu_score_customer'] = bleu_score_customer*100
        logs['bleu_score_agent'] = bleu_score_agent*100
        return logs

class PPLEvaluator(Evaluator):
    def __init__(self) -> None:
        super(PPLEvaluator, self).__init__()
    
    def evaluate(self, model: GPT2LMBase, scenes: List[Scene]) -> Optional[Dict[str, Any]]:
        _, loss_logs, _ = model.get_loss(scenes)
        logs = {}
        if 'token_loss' in loss_logs:
            logs['loss'] = loss_logs['token_loss']
        else:
            logs['loss'] = loss_logs['loss']
        return logs
    
    def postproc(self, logs: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        logs['ppl'] = math.exp(logs['evaluation']['loss'])
        return logs

