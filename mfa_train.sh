#!/usr/bin/env bash

dataset_root=${1:-"/home/konstantin/datasets"}
method=${2:-"fa"}
num_components=${3:-256}

rm "nohup.out"

nohup python3 -u mfa_train.py --dataset_root="${dataset_root}" --method="${method}" --num_components=${num_components} &
