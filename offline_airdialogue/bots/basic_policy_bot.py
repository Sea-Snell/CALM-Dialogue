import torch
from bots.base_bot import BaseBot
from ad.airdialogue import Event, Scene
from typing import Optional
from models.base import Policy

class BasicPolicyBot(BaseBot):
    def __init__(self, policy: Policy, **generation_kwargs) -> None:
        super(BasicPolicyBot, self).__init__()
        self.policy = policy
        self.generation_kwargs = generation_kwargs

    def respond(self, curr_event: Optional[Event], scene: Scene) -> Event:
        with torch.no_grad():
            generations, _ = self.policy.generate_one([scene], 
                                                      [curr_event.get_events()] if curr_event is not None else [[]], 
                                                      **self.generation_kwargs)
        return generations[0][0]
    
    def eval(self):
        self.policy.eval()
    
    def train(self):
        self.policy.train()