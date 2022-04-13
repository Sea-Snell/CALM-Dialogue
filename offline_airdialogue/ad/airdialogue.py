from __future__ import annotations
from typing import List, Optional, Dict, Any, Union, Set
from ad.ad_types import *
import json
from tqdm.auto import tqdm
import pickle as pkl
from collections import defaultdict
from dataclasses import dataclass
from ad.rewards import exact_match_reward, zero_reward
from utils.data_utils import DiscreteFeatures, discretize_price
from utils.misc import stack_dicts
from ad.heauristic_filters import heauristic_filter

@dataclass
class CustomerScenario:
    intention: Dict[str, Any]
    scene: Scene

    @classmethod
    def from_json(cls, val: Any, scene: Scene) -> CustomerScenario:
        return cls(
            intention=val['customer_scenario'],
            scene=scene,
        )
    
    def get_state(self):
        return self.intention

    def get_discrete_state(self, discrete_features: DiscreteFeatures):
        assert discrete_features is not None
        state = self.get_state().copy()
        state.pop('name')
        return discrete_features.encode(state)

    def get_str_state(self):
        keys = [
            'goal', 'name', 'max_price', 'max_connections', 'class',
            'airline_preference', 'departure_airport', 'departure_month',
            'departure_day', 'departure_time', 'return_airport', 'return_month',
            'return_day', 'return_time'
        ]
        state = self.get_state()
        return ' , '.join([k + ' ' + str(state[k]).strip() for k in keys])

    def __str__(self):
        return f"intention: {self.get_str_state()}"

@dataclass
class AgentScenario:
    kb: Dict[str, Any]
    scene: Scene

    @classmethod
    def from_json(cls, val: Any, scene: Scene) -> AgentScenario:
        return cls(
            kb=val['agent_scenario'],
            scene=scene,
        )

    def get_state(self, discrete_price=True):
        # add kind feature
        kb_copy = [dict(item) for item in self.kb['kb']]
        for i in range(len(kb_copy)):
            kb_copy[i]['kind'] = 'flight'
        # add price rank feature
        price_bins = defaultdict(list)
        for i, item in enumerate(kb_copy):
            price_bins[item['price']].append(i)
        for rank, (_, idxs) in enumerate(sorted(price_bins.items(), key=lambda z: z[0])):
            for idx in idxs:
                kb_copy[idx]['price_rank'] = rank
        # add empty row to beginning of table
        no_flight_dict = {k: 'None' for k in kb_copy[0].keys()}
        no_flight_dict['kind'] = 'flight'
        kb_copy = [no_flight_dict] + kb_copy
        # index flight numbers
        idx2_flight_num = list(map(lambda x: x['flight_number'], kb_copy))
        flight_num2_idx = {num: i for i, num in enumerate(idx2_flight_num)}
        # add reservation row to end of table
        reservation = self.kb['reservation']
        if reservation == 0:
            reservation = 'None'
        reservation_info = dict(kb_copy[flight_num2_idx[reservation]])
        reservation_info['kind'] = 'reservation'
        kb_copy = kb_copy + [reservation_info]
        # discretize price
        if discrete_price:
            for row in kb_copy:
                row['price'] = discretize_price(row['price'])

        return kb_copy
        

    def get_discrete_state(self, discrete_features: DiscreteFeatures):
        state = self.get_state()
        return stack_dicts([discrete_features.encode(row) for row in state])

    def _str_state_row(self, row, flat=False):
        keys = [
          'price', 'num_connections', 'class', 'airline', 'departure_airport',
          'departure_month', 'departure_day', 'departure_time_num',
          'return_airport', 'return_month', 'return_day', 'return_time_num'
        ]
        if flat:
            return f'{row["kind"]} {row["flight_number"]} , {" , ".join([str(row[k]).strip() for k in keys])}'
        else:
            return f'{row["kind"]} {row["flight_number"]} , {" , ".join([k.replace("num_", "").replace("_num", "") + " " + str(row[k]).strip() for k in keys])}'
    
    def get_str_state(self, flat=False):
        state = self.get_state(discrete_price=False)
        return [self._str_state_row(row, flat=flat) for row in state]

    def __str__(self, flat=False):
        return '\n'.join(self.get_str_state(flat=flat))

ScenarioType = Union[CustomerScenario, AgentScenario]

@dataclass
class Event:
    agent: Agent
    event: EventType
    event_id: str
    next_event: Optional[Event]
    prev_event: Optional[Event]
    scene: Scene
    data: Dict[str, Any]

    @classmethod
    def from_json(cls, val: Any, scene: Scene, idx: int=-1) -> Event:
        if val["action"] == "message":
            event = Message(val["data"])
        elif val["action"] == "invalid_event":
            event = InvalidEvent(val["data"])
        else:
            event = event_from_json(val["data"])
        return cls(
            agent=Agent(val["agent"]),
            event=event,
            event_id=f"{scene.scene_id}_{idx}",
            next_event=None,
            prev_event=None,
            scene=scene,
            data=dict(),
        )

    @classmethod
    def from_sentence(cls, speaker: str, s: str, scene: Scene, idx: int=-1):
        val = {}
        if speaker.lower() == 'submit':
            val['agent'] = 'Submit'
            try:
                event_str, name, flight = list(map(lambda x: x.strip(), s.split(',')))[:3]
                flight = int(flight)
                if (flight < 1000 or flight >= 1030) and flight != 0:
                    raise Exception
                if event_str not in {'book', 'no_flight', 'no_reservation', 'cancel', 'change'}:
                    raise Exception
                val['data'] = {'name': name, 'flight': [flight], 'status': event_str}
                val['action'] = event_str
            except Exception as e:
                val['data'] = s
                val['action'] = "invalid_event"
        else:
            val['data'] = s
            val['action'] = 'message'
            if speaker.lower() == 'agent':
                val['agent'] = 'Agent'
            elif speaker.lower() == 'customer':
                val['agent'] = 'Customer'
            else:
                val['agent'] = 'Submit'
                val['action'] = "invalid_event"
        return cls.from_json(val, scene, idx)

    def get_events(self, direction="prev") -> List[Event]:
        if direction == "prev":
            func = lambda ev: ev.prev_event
        elif direction == "next":
            func = lambda ev: ev.next_event
        else:
            raise NotImplementedError
        events = []
        ev = self
        while ev is not None:
            events.append(ev)
            ev = func(ev)
        if direction == 'prev':
            events.reverse()
        return events

    def get_all_events(self):
        return self.get_events() + self.get_events('next')[1:]

    def get_speaker(self):
        return str(self.agent)

    def is_final_action(self):
        return not isinstance(self.event, Message)
    
    def append(self, new_event: Event, link_forward=False):
        new_event.prev_event = self
        if link_forward:
            self.next_event = new_event
        return new_event
    
    def em_reward(self, expected_action: Optional[Dict[str, Any]] = None, check_name = True):
        if expected_action is None:
            expected_action = self.scene.expected_action
        if not (isinstance(self.event, InvalidEvent) or isinstance(self.event, Message)):
            return exact_match_reward(self.event.to_json(), 
                                      expected_action, 
                                      check_name=check_name)
        else:
            return zero_reward()


@dataclass
class Scene:
    customer_scenario: CustomerScenario
    agent_scenario: AgentScenario
    events: List[Event]
    scene_id: int
    expected_action: Dict[str, Any]
    data: Dict[str, Any]

    @classmethod
    def from_json(cls, val: Any) -> Scene:
        # create empty dummy scene
        scene = cls(None, None, [], val['uuid'], val['expected_action'], dict())
        # parse data
        scene.customer_scenario = CustomerScenario.from_json(val, scene)
        scene.agent_scenario = AgentScenario.from_json(val, scene)
        scene.events = [Event.from_json(ev, scene, i) for i, ev in enumerate(val["events"])]
        # link events
        for ev1, ev2 in zip(scene.events, scene.events[1:]):
            ev1.next_event = ev2
            ev2.prev_event = ev1
        # link to scene
        scene.customer_scenario.scene = scene
        scene.agent_scenario.scene = scene
        for ev in scene.events:
            ev.scene = scene
        return scene


class AirDialogue:
    def __init__(self, filepath: str, limit=None,
                 heauristic_filter=False, filter_with_goal=False):
        """Load a air dialogue dataset, along with addition event_data. Event
        data is a mapping from name to filepath, and filepath is a
        pickle file that contains a mapping from event_id to data.
        """
        self.heauristic_filter = heauristic_filter
        self.filter_with_goal = filter_with_goal
        self.scenes = self._load_in_scenes(filepath, limit=limit)
        self.events = {event.event_id: event for scene in self.scenes for event in scene.events}
        self.scenes_by_id = {scene.scene_id: scene for scene in self.scenes}

    def _load_in_scenes(self, filepath: str, limit=None) -> List[Scene]:
        scenes = []
        with open(filepath, "r") as f:
            for i, line in tqdm(enumerate(f)):
                if limit is not None and i >= limit:
                    break
                item = Scene.from_json(json.loads(line.strip()))
                if not (self.heauristic_filter and heauristic_filter(item, filter_with_goal=self.filter_with_goal)):
                    scenes.append(item)
        return scenes

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, i):
        return self.scenes[i]
