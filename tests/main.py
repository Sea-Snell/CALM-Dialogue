from ad.query_table import get_true_action
from ad.synthetic_iterator import AirDialogueIterator
from utils.data_utils import DiscreteFeatures
from ad.airdialogue import AirDialogue
from ad.ad_types import *
from collections import defaultdict

def get_ad_stats(ad):
    counts = defaultdict(int)
    for item in ad:
        if isinstance(item.events[-1].event, Book):
            counts['book'] += 1
        elif isinstance(item.events[-1].event, NoFlightFound):
            counts['no_flight'] += 1
        elif isinstance(item.events[-1].event, NoReservation):
            counts['no_reservation'] += 1
        elif isinstance(item.events[-1].event, Cancel):
            counts['cancel'] += 1
        elif isinstance(item.events[-1].event, Change):
            counts['change'] += 1
        else:
            raise NotImplementedError
        counts['reward'] += item.events[-1].em_reward()['reward']
        counts['flight_reward'] += item.events[-1].em_reward()['flight']
        counts['status_reward'] += item.events[-1].em_reward()['status']
        counts['name_reward'] += item.events[-1].em_reward()['name']
    return {**{k: (v / len(ad), v) for k, v in counts.items()}, 'n': len(ad)}

if __name__ == "__main__":
    # agent_features = DiscreteFeatures.from_file('../data/processed_ad/discrete_features.pkl', key='agent', strict=True)
    # customer_features = DiscreteFeatures.from_file('../data/processed_ad/discrete_features.pkl', key='customer', strict=True)
    ad = AirDialogue('../data/processed_ad/train_unfiltered.json', heauristic_filter=False, filter_with_goal=False)
    # print(get_ad_stats(ad))
    lens = []
    for item in ad:
        lens.append(len(item.events))
    print(min(lens))
    print(max(lens))
    print(sum(lens) / len(lens))
    

    # model = TableGPT2Agent(agent_features, device="cpu")
    # print(model.generate_one([ad[0].agent_scenario], [ad[0].events[:-4]], max_generation_len=32, num_generations=5))
    # synthetic = AirDialogueIterator('../data/raw_ad')
    # for i, item in enumerate(synthetic):
    #     print(i)
    #     assert item.expected_action == get_true_action(item.agent_scenario.kb['kb'], item.agent_scenario.kb['reservation'], item.customer_scenario.intention)
