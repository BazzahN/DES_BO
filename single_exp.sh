#!/bin/bash

CONFIG=$1
MACRO=$2

echo "================================="
echo " Running experiment: $CONFIG for m= $MACRO"
echo "================================="


echo "Running experiment script"
conda run --no-capture-output -n que python -u exp_script_single.py --macro $MACRO --config "configs/$CONFIG" | tee "$CONFIG".log
echo "done"
