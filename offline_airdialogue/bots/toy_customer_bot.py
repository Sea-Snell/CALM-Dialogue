from bots.base_bot import BaseBot
from ad.airdialogue import Event, Scene
from typing import Optional
from utils.sampling_utils import parse_utterance

class ToyCustomerBot(BaseBot):
    def __init__(self) -> None:
        super(ToyCustomerBot, self).__init__()
    
    def respond(self, curr_event: Optional[Event], scene: Scene) -> Event:
        return parse_utterance('Customer: ' + scene.customer_scenario.get_str_state(), scene, curr_event, link_forward=False)