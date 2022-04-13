from tqdm.auto import tqdm
import json
import argparse

def is_sample_valid(example, kb):
	return example['correct_sample']

def load_from_path(data_path, kb_path, limit=None, filter_invalid=False):
	keys = [
		'goal', 'name', 'max_price', 'max_connections', 'class',
		'airline_preference', 'departure_airport', 'departure_month',
		'departure_day', 'departure_time', 'return_airport', 'return_month',
		'return_day', 'return_time'
	]
	examples = []
	with open(data_path) as f_data, open(kb_path) as f_kb:
		for line_data, line_kb in tqdm(list(zip(f_data, f_kb))):
			example = json.loads(line_data)
			kb = json.loads(line_kb)
			if filter_invalid and (not is_sample_valid(example, kb)):
				continue
			new_dialogue = []
			for s in example['dialogue']:
				parts = s.split(': ')
				customer = parts[0]
				text = ': '.join(parts[1:])
				new_dialogue.append((customer, text))
			example['dialogue'] = new_dialogue

			intent = example['intent']
			for k in keys:
				if k not in intent:
					intent[k] = 'None'
			assert set(intent.keys()) == set(keys), f'{intent}'
			example['customer_state'] = intent
			example['agent_state'] = kb
			example.pop('search_info', None)
			example.pop('timestamps', None)
			example.pop('correct_sample', None)
			example.pop('intent', None)
			example.pop('kb', None)
			examples.append(example)
			if limit is not None and len(examples) >= limit:
				break
	return examples

def factor_into_events(id, interaction):
	event_data = {}
	event_data['uuid'] = id
	event_data['customer_scenario_uuid'] = 2*id
	event_data['agent_scenario_uuid'] = 2*id+1
	event_data['customer_scenario'] = interaction['customer_state']
	event_data['agent_scenario'] = interaction['agent_state']
	event_data['expected_action'] = interaction['expected_action']
	event_data['events'] = []
	for agent, data in interaction['dialogue']:
		if agent == 'customer':
			agent = 'Customer'
		elif agent == 'agent':
			agent = 'Agent'
		else:
			raise NotImplementedError
		event_data['events'].append({
			'agent': agent,
			'data': data,
			'action': 'message',
		})
	event_data['events'].append({
		'agent': 'Submit',
		'data': interaction['action'],
		'action': interaction['action']['status']
	})
	return event_data

def main(data_file, kb_file, out_file, limit=None, filter_invalid=False):
	print('loading data...')
	items = load_from_path(data_file, kb_file, limit=limit, filter_invalid=filter_invalid)
	print('reformatting data...')
	data = []
	for i, item in tqdm(enumerate(items)):
		data.append(factor_into_events(i, item))
	print('saving data...')
	with open(out_file, 'w') as f:
		f.write('\n'.join([json.dumps(item) for item in data]))

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_file")
	parser.add_argument("--kb_file")
	parser.add_argument("--out_file")
	parser.add_argument("--limit", type=int, default=None)
	parser.add_argument("--filter_invalid", action='store_true')
	args = parser.parse_args()
	main(args.data_file, args.kb_file, args.out_file, limit=args.limit, filter_invalid=args.filter_invalid)