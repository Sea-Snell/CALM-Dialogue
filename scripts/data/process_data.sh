#!/bin/bash

python preprocess_data.py \
        --data_file ../../data/raw_ad/airdialogue/train_data.json \
        --kb_file ../../data/raw_ad/airdialogue/train_kb.json \
        --out_file ../../data/processed_ad/train_unfiltered.json

python preprocess_data.py \
        --data_file ../../data/raw_ad/airdialogue/train_data.json \
        --kb_file ../../data/raw_ad/airdialogue/train_kb.json \
        --out_file ../../data/processed_ad/train_filtered.json \
        --filter_invalid

python preprocess_data.py \
        --data_file ../../data/raw_ad/airdialogue/dev_data.json \
        --kb_file ../../data/raw_ad/airdialogue/dev_kb.json \
        --out_file ../../data/processed_ad/dev_unfiltered.json

python preprocess_data.py \
        --data_file ../../data/raw_ad/airdialogue/dev_data.json \
        --kb_file ../../data/raw_ad/airdialogue/dev_kb.json \
        --out_file ../../data/processed_ad/dev_filtered.json \
        --filter_invalid



python preprocess_data.py \
        --data_file ../../data/raw_ad/airdialogue/train_data.json \
        --kb_file ../../data/raw_ad/airdialogue/train_kb.json \
        --out_file ../../data/processed_ad/train_unfiltered_small.json \
        --limit 1000

python preprocess_data.py \
        --data_file ../../data/raw_ad/airdialogue/train_data.json \
        --kb_file ../../data/raw_ad/airdialogue/train_kb.json \
        --out_file ../../data/processed_ad/train_filtered_small.json \
        --filter_invalid \
        --limit 1000

python preprocess_data.py \
        --data_file ../../data/raw_ad/airdialogue/dev_data.json \
        --kb_file ../../data/raw_ad/airdialogue/dev_kb.json \
        --out_file ../../data/processed_ad/dev_unfiltered_small.json \
        --limit 1000

python preprocess_data.py \
        --data_file ../../data/raw_ad/airdialogue/dev_data.json \
        --kb_file ../../data/raw_ad/airdialogue/dev_kb.json \
        --out_file ../../data/processed_ad/dev_filtered_small.json \
        --filter_invalid \
        --limit 1000


python discretize_data.py --data_files ../../data/processed_ad/train_unfiltered.json \
                                       ../../data/processed_ad/dev_unfiltered.json \
                          --out_file ../../data/processed_ad/discrete_features.pkl