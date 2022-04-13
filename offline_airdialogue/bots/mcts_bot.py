from collections import defaultdict
import torch
from bots.base_bot import BaseBot
from ad.airdialogue import Event, Scene
from typing import Optional
from models.base import Policy, RewardFunction

class MCTSBot(BaseBot):
    def __init__(self, policy: Policy, reward: RewardFunction, generation_kwargs, reward_kwargs) -> None:
        super(MCTSBot, self).__init__()
        self.policy = policy
        self.reward = reward
        self.generation_kwargs = generation_kwargs
        self.reward_kwargs = reward_kwargs

    def respond(self, curr_event: Optional[Event], scene: Scene) -> Event:
        with torch.no_grad():
            generations, _ = self.policy.generate_full([scene], 
                                                       [curr_event.get_events()] if curr_event is not None else [[]], 
                                                       **self.generation_kwargs)
            rewards = self.reward.get_reward([scene for _ in generations[0]], 
                                             [generation.get_events() for generation in generations[0]], 
                                             **self.reward_kwargs)
        start_idx = len(curr_event.get_events()) if curr_event is not None else 0
        next_utterances = [str(generation.get_events()[start_idx].event) for generation in generations[0]]
        next_utterance2idx = {utterance: i for i, utterance in enumerate(next_utterances)}
        next_utterance2reward = defaultdict(lambda: [0, 0])
        for i, utterance in enumerate(next_utterances):
            next_utterance2reward[utterance][0] += rewards[i]
            next_utterance2reward[utterance][1] += 1
        best_utterance = max(next_utterance2reward.keys(), key=lambda k: next_utterance2reward[k][0] / next_utterance2reward[k][1])
        return generations[0][next_utterance2idx[best_utterance]].get_events()[start_idx]
    
    def eval(self):
        self.policy.eval()
    
    def train(self):
        self.policy.train()