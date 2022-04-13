import argparse
import random
import json

def main(processed_data_file, out_file, n, seed):
    data = [json.loads(item.strip()) for item in open(processed_data_file, 'r')]
    assert list(map(lambda x: x['uuid'], data)) == sorted(list(map(lambda x: x['uuid'], data)))
    random.seed(seed)
    idxs = random.sample(range(len(data)), n)
    with open(out_file, 'w') as f:
        f.write('\n'.join([json.dumps(data[idx]) for idx in idxs]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_data_file", type=str)
    parser.add_argument("--out_file", type=str)
    parser.add_argument("--n", type=int)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.processed_data_file, args.out_file, args.n, args.seed)