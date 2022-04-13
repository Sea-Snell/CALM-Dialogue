from os import link
from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F
from ad.ad_types import Agent
from ad.airdialogue import Event, Scene
from utils.misc import strip_from_beginning, strip_from_end
import numpy as np

select_batch_idxs = lambda x, idxs: torch.gather(x, dim=0, index=idxs.repeat(*x.shape[1:], 1).permute(len(x.shape) - 1, *list(range(len(x.shape) - 1))))
map_all_kvs = lambda f, kvs: tuple([tuple(map(f, items)) for items in kvs])
map_decoder_kvs = lambda f, kvs: tuple([tuple(map(f, items[:2]))+tuple(items[2:]) for items in kvs])
pad_sequence = lambda seq, to_len, val, device, dim: torch.cat((seq, torch.full((*seq.shape[:dim], to_len-seq.shape[dim], *seq.shape[(dim+1):]), val).to(device)), dim=dim)

def update_kvs(kvs, updated_kvs, lens_chosen, idx):
    for i, layer in enumerate(kvs):
        for x, item in enumerate(layer):
            item[lens_chosen, :, idx, :] = updated_kvs[i][x][:, :, idx, :]
    return kvs

def update_decoder_kvs(kvs, updated_kvs, lens_chosen, idx):
    for i, layer in enumerate(kvs):
        for x, item in enumerate(layer[:2]):
            item[lens_chosen, :, idx, :] = updated_kvs[i][x][:, :, idx, :]
    return kvs

def get_relevent_kvs(kvs, lens_chosen, idx):
    kvs = map_all_kvs(lambda x: select_batch_idxs(x, lens_chosen), kvs)
    kvs = map_all_kvs(lambda x: x[:,:,:idx,:], kvs)
    return kvs

def top_k_logits(logits, k):
    # logits = (batch, time, dim)
    _, bottom_k_idx = torch.topk(-logits, logits.shape[2]-k, dim=2)
    return torch.scatter(logits, dim=2, index=bottom_k_idx, value=float('-inf'))

def top_p_logits(logits, p):
    # logits = (batch, time, dim)
    sorted_logits, _ = torch.sort(logits, dim=2, descending=True)
    num_to_take = torch.sum(torch.cumsum(F.softmax(sorted_logits, dim=2), dim=2) <= p, dim=2).unsqueeze(2)
    mask = logits < torch.gather(sorted_logits, dim=2, index=torch.clamp(num_to_take, max=logits.shape[2]-1))
    return logits.masked_fill(mask, float('-inf'))

def process_logits(logits, temp=1.0, top_k=None, top_p=None):
    logits /= temp
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    if top_p is not None:
        logits = top_p_logits(logits, top_p)
    return logits

def single_utterance_termination_condition(dialogue):
    return True

def end_of_dialogue_termination_condition(dialogue, sep_token, final_speaker_str):
    return any(list(map(lambda x: x.strip().lower().startswith(final_speaker_str.lower()), dialogue.split(sep_token))))

def parse_utterance(utterance: str, scene: Scene, prev_event: Optional[Event], link_forward: bool = False):
    utterance = utterance.strip()
    utterance = strip_from_end(utterance, '</s>').strip()
    utterance = strip_from_beginning(utterance, '<s>').strip()
    for agent in Agent:
        if utterance.startswith(f'{str(agent)}:'):
            event = Event.from_sentence(str(agent), strip_from_beginning(utterance, f'{str(agent)}:').strip(), scene)
            return prev_event.append(event, link_forward=link_forward) if prev_event is not None else event
    event = Event.from_sentence('ERROR', utterance, scene)
    return prev_event.append(event, link_forward=link_forward) if prev_event is not None else event

def parse_utterances(utterances: str, scene: Scene, prev_event: Optional[Event], link_forward: bool = False):
    curr_event = prev_event
    for utterance in utterances.split('</s>'):
        if len(utterance.strip()) == 0:
            continue
        curr_event = parse_utterance(utterance.strip(), scene, curr_event, link_forward=link_forward)
    return curr_event

def select_max_prob_response(generations: Tuple[str, List[str]], probs: np.ndarray):
    samples = generations[1]
    sample = samples[np.argmax(probs)]
    return sample