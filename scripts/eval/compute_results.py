import argparse
import json

def avg_reward(items, metric):
    total = 0.0
    for item in items:
        total += metric(item)
    return total / len(items)

def accuracy_stats(items):
    reward_metrics = {'reward_accuracy': lambda x: float(x['reward']['reward'] == 1.0),
                     'status_accuracy': lambda x: float(x['reward']['status'] == 1.0),
                     'flight_accuracy': lambda x: float(x['reward']['flight'] == 1.0),
                     'name_accuracy': lambda x: float(x['reward']['name'] == 1.0),
                     }
    stats = {}
    for k, v in reward_metrics.items():
        stats[k] = avg_reward(items, v)
    stats['n'] = len(items)
    return stats

def main(results_file):
    results = [json.loads(item) for item in open(results_file, 'r')]
    print(accuracy_stats(results))

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--results_file")
	args = parser.parse_args()
	main(args.results_file)