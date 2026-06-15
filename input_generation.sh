#!/bin/bash

CONFIG=$1
N_MACROS=$2

echo "================================="
echo " Generating Input: $CONFIG"
echo "================================="

conda run -n bo_prime python input_generation.py --config "configs/$CONFIG.yml" --n_macros $N_MACROS
echo "done"

