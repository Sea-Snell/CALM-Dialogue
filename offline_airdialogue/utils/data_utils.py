from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Set, Tuple
import pickle as pkl
from collections import defaultdict
import random

@dataclass
class DiscreteFeatures:
    val_sets: Dict[str, Set[Any]]
    idx2val: Dict[str, List[Any]]
    val2idx: Dict[str, Dict[Any, int]]
    strict: bool = False

    @classmethod
    def from_json(cls, discrete_features: Dict[str, Any], strict: bool=False) -> DiscreteFeatures:
        return cls(
            val_sets=discrete_features['val_sets'],
            idx2val=discrete_features['idx2val'],
            val2idx=discrete_features['val2idx'],
            strict=strict,
        )
    
    def to_json(self):
        return {'val_sets': self.val_sets, 'idx2val': self.idx2val, 'val2idx': self.val2idx}
    
    @classmethod
    def from_file(cls, path_to_discrete: str, key: Optional[str]=None, strict: bool=False):
        with open(path_to_discrete, 'rb') as f:
            discrete_features = pkl.load(f)
        return cls.from_json(discrete_features[key] if key is not None else discrete_features, strict=strict)
    
    @classmethod
    def from_data(cls, dicts: List[Dict[str, Any]], additional_vals: Optional[Dict[str, List[Any]]]=None, 
                  ignore_idx: Optional[Set[str]]=None, strict: bool=False) -> DiscreteFeatures:
        val_sets = defaultdict(set)
        for item in dicts:
            for k, v in item.items():
                if ignore_idx is not None and k in ignore_idx:
                    continue
                val_sets[k].add(v)
        if additional_vals is not None:
            for k, v in additional_vals.items():
                val_sets[k].update(v)
        idx2val = {k: sorted(list(v.difference({'None'})))+['None'] if 'None' in v else sorted(list(v)) for k, v in val_sets.items()}
        val2idx = {k: {v: idx for idx, v in enumerate(vals)} for k, vals in idx2val.items()}
        return cls(val_sets=val_sets, 
                   idx2val=idx2val, 
                   val2idx=val2idx,
                   strict=strict)
    
    def encode(self, items: Dict[str, Any]) -> Dict[str, int]:
        encoded = {}
        for k, v in items.items():
            if k not in self.val2idx:
                if self.strict:
                    raise Exception
                continue
            encoded[k] = self.val2idx[k][v]
        return encoded
    
    def decode(self, items: Dict[str, int]) -> Dict[str, Any]:
        decoded = {}
        for k, v in items.items():
            if k not in self.idx2val:
                if self.strict:
                    raise Exception
                continue
            decoded[k] = self.idx2val[k][v]
    
    def get_emb_spec(self):
        return {k: len(v) for k, v in self.val_sets.items()}

def discretize_price(price):
    if price == 'None':
        return 'None'
    if price <= 200:
        return 200
    if price <= 500:
        return 500
    if price <= 1000:
        return 1000
    return 5000

class ActionEnumerator:
    def __init__(self):
        self._idx2action = [('no_flight', 0,),
                            ('no_reservation', 0,),
                            ('cancel', 0,)]
        for i in range(1000, 1030):
            self._idx2action.append(('book', i,))
            self._idx2action.append(('change', i,))
        self._action2idx = {action: i for i, action in enumerate(self._idx2action)}
    
    def _action2tuple(self, action: Dict[str, Any]):
        assert len(action['flight']) <= 1
        if len(action['flight']) != 0:
            return (action['status'], action['flight'][0])
        return (action['status'], 0)
    
    def _tuple2action(self, action_tuple: Tuple[str, int]):
        if action_tuple[1] != 0:
            return {'status': action_tuple[0], 'flight': [action_tuple[1]]}
        return {'status': action_tuple[0], 'flight': []}

    def action2idx(self, taken_action: Dict[str, Any]):
        action_tuple = self._action2tuple(taken_action)
        return self._action2idx[action_tuple]
    
    def idx2action(self, idx: int):
        action_tuple = self._idx2action[idx]
        return self._tuple2action(action_tuple)
    
    def n_actions(self):
        return len(self._idx2action)
    
    def factor_expected_action(self, expected_action: Dict[str, Any]):
        valid_actions = []
        if len(expected_action['flight']) == 0:
            valid_actions.append({'status': expected_action['status'], 'flight': []})
        else:
            for flight in expected_action['flight']:
                valid_actions.append({'status': expected_action['status'], 'flight': [flight]})
        return valid_actions
    
    def sample_action_with_reward(self, expected_action: Dict[str, Any], reward: bool):
        correct_actions = self.factor_expected_action(expected_action)
        if reward:
            return random.choice(correct_actions)
        incorrect_actions = []
        for i in range(self.n_actions()):
            action = self.idx2action(i)
            if action not in correct_actions:
                incorrect_actions.append(action)
        return random.choice(incorrect_actions)
