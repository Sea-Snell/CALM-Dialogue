import pickle as pkl
from utils.data_utils import DiscreteFeatures
from ad.airdialogue import Scene
import json
import argparse

def discretize(data_items):
	agent_dicts = []
	customer_dicts = []
	for item in data_items:
		scene = Scene.from_json(item)
		agent_dicts.extend(scene.agent_scenario.get_state(discrete_price=True))
		customer_dicts.append(scene.customer_scenario.get_state())
	agent_features = DiscreteFeatures.from_data(agent_dicts, additional_vals={'price_rank': list(range(30))}, strict=False)
	customer_features = DiscreteFeatures.from_data(customer_dicts, ignore_idx={'name'}, strict=False)
	return customer_features.to_json(), agent_features.to_json()

def main(data_files, out_file):
    print(f'pulling from: {data_files}')
    data_items = []
    for data_file in data_files:
        print(f'loading: {data_file}')
        with open(data_file, 'r') as f:
            for item in f:
                data_items.append(json.loads(item.strip()))
    print('collecting discrete sets ...')
    customer_features, agent_features = discretize(data_items)
    print(f'saving to {out_file} ...')
    with open(out_file, 'wb') as f:
        pkl.dump({'customer': customer_features, 'agent': agent_features}, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_files", nargs='+', type=str)
    parser.add_argument("--out_file")
    args = parser.parse_args()
    main(args.data_files, args.out_file)