from __future__ import annotations
from typing import Union
from dataclasses import dataclass
import enum


class Agent(enum.Enum):
    AGENT = "Agent"
    CUSTOMER = "Customer"
    SUBMIT = "Submit"

    def other_agent(self) -> Agent:
        if self == self.CUSTOMER:
            return self.AGENT
        elif self == self.AGENT:
            return self.CUSTOMER
        elif self == self.SUBMIT:
            return self.SUBMIT
        else:
            raise NotImplementedError

    def __str__(self) -> str:
        if self == self.CUSTOMER:
            return "Customer"
        elif self == self.AGENT:
            return "Agent"
        elif self == self.SUBMIT:
            return "Submit"
        else:
            raise NotImplementedError

@dataclass
class Message:
    utterance: str

    def __str__(self):
        return self.utterance
    
    def kind_str(self):
        return 'message'


@dataclass
class Book:
    customer_name: str
    flight_selected: int

    def __str__(self):
        return f"book , {self.customer_name} , {self.flight_selected}"

    def to_json(self):
        return {'flight': [self.flight_selected], 'status': 'book', 'name': self.customer_name}
    
    def kind_str(self):
        return 'book'


@dataclass
class Change:
    customer_name: str
    flight_selected: int

    def __str__(self):
        return f"change , {self.customer_name} , {self.flight_selected}"

    def to_json(self):
        return {'flight': [self.flight_selected], 'status': 'change', 'name': self.customer_name}
    
    def kind_str(self):
        return 'change'


@dataclass
class NoFlightFound:
    customer_name: str

    def __str__(self):
        return f"no_flight , {self.customer_name} , 0"

    def to_json(self):
        return {'flight': [], 'status': 'no_flight', 'name': self.customer_name}
    
    def kind_str(self):
        return 'no_flight'


@dataclass
class NoReservation:
    customer_name: str

    def __str__(self):
        return f"no_reservation , {self.customer_name} , 0"

    def to_json(self):
        return {'flight': [], 'status': 'no_reservation', 'name': self.customer_name}
    
    def kind_str(self):
        return 'no_reservation'

@dataclass
class Cancel:
    customer_name: str

    def __str__(self):
        return f"cancel , {self.customer_name} , 0"

    def to_json(self):
        return {'flight': [], 'status': 'cancel', 'name': self.customer_name}
    
    def kind_str(self):
        return 'cancel'

@dataclass
class InvalidEvent:
    raw_text: str

    def __str__(self):
        return self.raw_text
    
    def to_json(self):
        return {'flight': None, 'status': None, 'name': None}
    
    def kind_str(self):
        return 'invalid_event'


EventType = Union[Message, Book, Change, NoFlightFound, NoReservation, Cancel, InvalidEvent]

def event_to_int(ev: EventType) -> int:
    if isinstance(ev, Message):
        return 0
    elif isinstance(ev, Book):
        return 1
    elif isinstance(ev, Change):
        return 2
    elif isinstance(ev, NoFlightFound):
        return 3
    elif isinstance(ev, NoReservation):
        return 4
    elif isinstance(ev, Cancel):
        return 5
    elif isinstance(ev, InvalidEvent):
        return 6
    else:
        raise NotImplementedError

def event_from_json(json_obj):
    if json_obj["status"] == "book":
        event = Book(json_obj["name"], json_obj["flight"][0])
    elif json_obj["status"] == "change":
        event = Change(json_obj["name"], json_obj["flight"][0])
    elif json_obj["status"] == "no_flight":
        event = NoFlightFound(json_obj["name"])
    elif json_obj["status"] == "no_reservation":
        event = NoReservation(json_obj["name"])
    elif json_obj["status"] == "cancel":
        event = Cancel(json_obj["name"])
    else:
        raise NotImplementedError
    return event