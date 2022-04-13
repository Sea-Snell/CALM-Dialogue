import os
from airdialogue.context_generator.context_generator_lib import ContextGenerator
from ad.airdialogue import Scene
import numpy as np
import random
from datetime import datetime
from airdialogue.context_generator.src import kb as knowledgebase
from ad.query_table import get_true_action

class AirDialogueIterator:
    def __init__(self, ad_dir: str) -> None:
        self.generator = ContextGenerator(num_candidate_airports=3,
                                                  book_window=2,
                                                  num_db_record=30,
                                                  firstname_file=os.path.join(ad_dir, 'resources/meta_context/first_names.txt'),
                                                  lastname_file=os.path.join(ad_dir, 'resources/meta_context/last_names.txt'),
                                                  airportcode_file=os.path.join(ad_dir, 'resources/meta_context/airport.txt'))
        self.keys = [
            'goal', 'name', 'max_price', 'max_connections', 'class',
            'airline_preference', 'departure_airport', 'departure_month',
            'departure_day', 'departure_time', 'return_airport', 'return_month',
            'return_day', 'return_time'
        ]

        self.months_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5,
                           'June': 6, 'July': 7, 'Aug': 8, 'Sept': 9,
                           'Oct': 10, 'Nov': 11, 'Dec': 12}

    def sample_scene(self) -> Scene:
        generated_item = self.generator.generate_context(1,
                                                    output_data=None,
                                                    output_kb=None,
                                                    display_freq=None,
                                                    verbose=False)[0][0]
        kb = {'kb': generated_item['kb'], 'reservation': generated_item['reservation']}
        intent = generated_item['intent']
        for k in self.keys:
            if k not in intent:
                intent[k] = 'None'
        assert set(intent.keys()) == set(self.keys), f'{intent}'
        generated_item['customer_state'] = intent
        generated_item['agent_state'] = kb
        generated_item.pop('search_info', None)
        generated_item.pop('timestamps', None)
        generated_item.pop('correct_sample', None)
        generated_item.pop('intent', None)
        generated_item.pop('kb', None)

        event_data = {}
        event_data['uuid'] = -1
        event_data['customer_senario_uuid'] = -1
        event_data['agent_senario_uuid'] = -1
        event_data['customer_scenario'] = generated_item['customer_state']
        event_data['agent_scenario'] = generated_item['agent_state']
        event_data['expected_action'] = generated_item['expected_action']
        event_data['events'] = []

        return Scene.from_json(event_data)
    
    def sample_scene_conditioned_on_intent(self, intent) -> Scene:
        airport_candidate = list(np.random.choice(
                                    self.generator.fact_obj.airport_list,
                                    self.generator.num_candidate_airports,
                                    replace=False))
        origin = random.randint(0, len(airport_candidate) - 1)
        dest = random.randint(0, len(airport_candidate) - 1)
        if dest == origin:
            dest = (dest + 1) % len(airport_candidate)
        if intent['return_airport'] not in airport_candidate:
            airport_candidate[dest] = intent['return_airport']
        if intent['departure_airport'] not in airport_candidate:
            airport_candidate[origin] = intent['departure_airport']
        start_time = datetime.utcfromtimestamp(self.generator.fact_obj.base_departure_time_epoch)
        start_time = start_time.replace(hour=0, minute=0, second=0)
        end_time = datetime.utcfromtimestamp(start_time.timestamp() + 24*3600)
        sampled_time = datetime.utcfromtimestamp(random.randint(start_time.timestamp(), end_time.timestamp()))
        departure_date = sampled_time.replace(month=self.months_map[intent['departure_month']], day=int(intent['departure_day']))
        if departure_date.timestamp() < sampled_time.timestamp():
            departure_date = departure_date.replace(year=sampled_time.year+1)
        return_date = datetime.utcfromtimestamp(departure_date.timestamp() + 3600*24*self.generator.book_window)
        kb = knowledgebase.Knowledgebase(self.generator.fact_obj, self.generator.num_db_record,
                                         airport_candidate, departure_date.timestamp(),
                                         return_date.timestamp()).get_json()
        kb = {'kb': kb['kb'], 'reservation': kb['reservation']}
        expected_action = get_true_action(kb['kb'], kb['reservation'], intent)
        for k in self.keys:
            if k not in intent:
                intent[k] = 'None'
        assert set(intent.keys()) == set(self.keys), f'{intent}'
        event_data = {}
        event_data['uuid'] = -1
        event_data['customer_senario_uuid'] = -1
        event_data['agent_senario_uuid'] = -1
        event_data['customer_scenario'] = intent
        event_data['agent_scenario'] = kb
        event_data['expected_action'] = expected_action
        event_data['events'] = []

        return Scene.from_json(event_data)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        return self.sample_scene()
    
    