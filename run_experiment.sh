#!/bin/bash

CONFIG=$1

echo "================================="
echo " Running experiment: $CONFIG"
echo "================================="

echo "Generating Input"
conda run -n que python input_generation.py --config "configs/$CONFIG"
echo "done"

echo "Running experiment script"
conda run --no-capture-output -n que python -u exp_script.py --config "configs/$CONFIG" | tee "$CONFIG".log
echo "done"
