#!/usr/bin/env bash

dataset_root=${1:-"/home/konstantin/datasets"}
method=${2:-"ppca"}
num_components=${3:-256}
model=${4:-"/home/konstantin/personal/OnGansAndGMMs/run/e5_ppca_256_saved_gmm.pkl"}

rm "nohup.out"

nohup python3 -u mfa_train.py --dataset_root="${dataset_root}" --method="${method}" --num_components=${num_components} --model="${model}" &
