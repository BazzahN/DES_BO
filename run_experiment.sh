#!/bin/bash

CONFIG=$1
MACROS=$2
echo "================================="
echo " Running experiment: $CONFIG |M=$MACROS"
echo "================================="

echo "Running experiment script"
conda run --no-capture-output -n bo_prime python -u exp_script.py --config "configs/$CONFIG.yml" --n_macros $MACROS | tee "logs/$CONFIG".log
echo "done"
#Notify on completion
notify-send "Experiment $Config Complete"
