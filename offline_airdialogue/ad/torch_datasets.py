from numpy.lib.type_check import real
from torch.utils.data import IterableDataset, Dataset
from ad.synthetic_iterator import AirDialogueIterator
from utils.data_utils import ActionEnumerator
import random
from ad.airdialogue import Event, event_from_json, AirDialogue, Scene
from ad.query_table import get_true_action
from copy import deepcopy

class BasicDataset(Dataset):
    def __init__(self, data: AirDialogue, cond_reward_key: str, cond_reward: float):
        self.data = data
        self.cond_reward_key = cond_reward_key
        self.cond_reward = cond_reward
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        scene = self.data[i]
        scene.data[self.cond_reward_key] = self.cond_reward
        return scene

    @staticmethod
    def collate(items):
        return items

class ToySynthTableDataset(IterableDataset):
    def __init__(self, data: AirDialogueIterator, cond_reward_key: str, 
                 reward_1_prob: float = 0.5):
        self.data = data
        self.cond_reward_key = cond_reward_key
        self.reward_1_prob = reward_1_prob
        self.action_enum = ActionEnumerator()

    def __iter__(self):
        return self

    def __next__(self):
        scene = next(self.data)
        customer_event = Event.from_sentence('Customer', scene.customer_scenario.get_str_state(), scene)
        reward = float(random.random() < self.reward_1_prob)
        temp_action = self.action_enum.sample_action_with_reward(scene.expected_action, reward)
        temp_action['name'] = scene.expected_action['name']
        action_event = Event.from_sentence('Submit', str(event_from_json(temp_action)), scene)
        scene.events = customer_event.append(action_event).get_events()
        scene.data[self.cond_reward_key] = reward
        return scene

    @staticmethod
    def collate(items):
        return items

class ToySynthTableDataset2(IterableDataset):
    def __init__(self, data: AirDialogueIterator, cond_reward_key: str):
        self.data = data
        self.cond_reward_key = cond_reward_key
        self.action_enum = ActionEnumerator()

    def __iter__(self):
        return self

    def __next__(self):
        scene = next(self.data)
        customer_event = Event.from_sentence('Customer', scene.customer_scenario.get_str_state(), scene)
        temp_action = self.action_enum.sample_action_with_reward(scene.expected_action, 1.0)
        temp_action['name'] = scene.expected_action['name']
        action_event = Event.from_sentence('Submit', str(event_from_json(temp_action)), scene)
        scene.events = customer_event.append(action_event).get_events()
        scene.data[self.cond_reward_key] = 1.0
        return scene

    @staticmethod
    def collate(items):
        return items

class RealSynthTableDataset(IterableDataset):
    def __init__(self, synth_data: AirDialogueIterator, real_data: AirDialogue, 
                 cond_reward_key: str, max_retries: int, original_prob: float, verbose: bool = True, 
                 reward_1_prob: float = 0.5):
        self.synth_data = synth_data
        self.real_data = real_data
        self.cond_reward_key = cond_reward_key
        self.max_retries = max_retries
        self.original_prob = original_prob
        self.verbose = verbose
        self.reward_1_prob = reward_1_prob
        self.action_enum = ActionEnumerator()

    def __iter__(self):
        return self

    def _choose_scene(self, original_scene):
        if random.random() < self.original_prob:
            return original_scene
        try:
            return self.synth_data.sample_scene_conditioned_on_intent(original_scene.customer_scenario.intention)
        except Exception as e:
            if self.verbose:
                print(e)
            return original_scene
    
    def _sample_scene_with_reward(self, real_scene: Scene, reward: float):
        original_scene = deepcopy(real_scene)
        scene = self._choose_scene(original_scene)
        scene.agent_scenario.scene = real_scene
        real_scene.agent_scenario = scene.agent_scenario
        true_action = get_true_action(real_scene.agent_scenario.kb['kb'], 
                                      real_scene.agent_scenario.kb['reservation'], 
                                      real_scene.customer_scenario.intention)
        real_scene.expected_action = true_action
        retries = 0
        while real_scene.events[-1].em_reward()['reward'] != reward and retries < self.max_retries:
            scene = self._choose_scene(original_scene)
            scene.agent_scenario.scene = real_scene
            real_scene.agent_scenario = scene.agent_scenario
            true_action = get_true_action(real_scene.agent_scenario.kb['kb'], 
                                          real_scene.agent_scenario.kb['reservation'], 
                                          real_scene.customer_scenario.intention)
            real_scene.expected_action = true_action
            retries += 1
        real_scene.data[self.cond_reward_key] = real_scene.events[-1].em_reward()['reward']
        return real_scene
    
    def __next__(self):
        real_scene = deepcopy(random.choice(self.real_data))
        reward = float(random.random() < self.reward_1_prob)
        sampled_scene = self._sample_scene_with_reward(real_scene, reward)
        return sampled_scene

    @staticmethod
    def collate(items):
        return items


class RealSynthTableDataset2(IterableDataset):
    def __init__(self, synth_data: AirDialogueIterator, real_data: AirDialogue, 
                 cond_reward_key: str, max_retries: int, original_prob: float, verbose: bool = True):
        self.synth_data = synth_data
        self.real_data = real_data
        self.cond_reward_key = cond_reward_key
        self.max_retries = max_retries
        self.original_prob = original_prob
        self.verbose = verbose
        self.action_enum = ActionEnumerator()

    def __iter__(self):
        return self

    def _choose_scene(self, original_scene):
        if random.random() < self.original_prob:
            return original_scene
        try:
            return self.synth_data.sample_scene_conditioned_on_intent(original_scene.customer_scenario.intention)
        except Exception as e:
            if self.verbose:
                print(e)
            return original_scene
    
    def _sample_scene_with_reward(self):
        while True:
            real_scene = deepcopy(random.choice(self.real_data))
            original_scene = deepcopy(real_scene)
            scene = self._choose_scene(original_scene)
            scene.agent_scenario.scene = real_scene
            real_scene.agent_scenario = scene.agent_scenario
            true_action = get_true_action(real_scene.agent_scenario.kb['kb'], 
                                        real_scene.agent_scenario.kb['reservation'], 
                                        real_scene.customer_scenario.intention)
            real_scene.expected_action = true_action
            retries = 0
            while real_scene.events[-1].em_reward()['reward'] != 1.0 and retries < self.max_retries:
                scene = self._choose_scene(original_scene)
                scene.agent_scenario.scene = real_scene
                real_scene.agent_scenario = scene.agent_scenario
                true_action = get_true_action(real_scene.agent_scenario.kb['kb'], 
                                            real_scene.agent_scenario.kb['reservation'], 
                                            real_scene.customer_scenario.intention)
                real_scene.expected_action = true_action
                retries += 1
            if real_scene.events[-1].em_reward()['reward'] == 1.0:
                break
        real_scene.data[self.cond_reward_key] = real_scene.events[-1].em_reward()['reward']
        return real_scene
    
    def __next__(self):
        sampled_scene = self._sample_scene_with_reward()
        return sampled_scene

    @staticmethod
    def collate(items):
        return items
