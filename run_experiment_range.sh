#!/bin/bash

CONFIG=$1
MACROS=$2
M_min=$3
echo "================================="
echo " Running experiment: $CONFIG |M=$MACROS"
echo "================================="

echo "Running experiment script"
conda run --no-capture-output -n bo_prime python -u exp_script.py --config "configs/$CONFIG.yml" --n_macros $MACROS --m_min $M_min | tee "logs/$CONFIG".log
echo "done"
