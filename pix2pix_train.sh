#!/usr/bin/env bash

dataset_root=${1:-"/home/konstantin/datasets"}
mfa_model_path=${2:-"/home/konstantin/personal/OnGansAndGMMs/run/e1_fa_256_saved_gmm.pkl"}

rm "nohup.out"

nohup python3 -u pix2pix_train.py --dataset_root="${dataset_root}" --mfa_model_path="${mfa_model_path}" &