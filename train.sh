#!/usr/bin/env bash

dataset_root=${1:-"/home/konstantin/datasets"}

rm "nohup.out"

nohup python3 -u mfa_train.py --dataset_root="${dataset_root}" &
