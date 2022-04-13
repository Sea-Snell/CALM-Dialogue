from abc import ABC, abstractmethod
from ad.airdialogue import Event, Scene
from typing import Optional

class BaseBot(ABC):
    @abstractmethod
    def respond(self, curr_event: Optional[Event], scene: Scene) -> Event:
        pass

    def eval(self):
        pass

    def train(self):
        pass

