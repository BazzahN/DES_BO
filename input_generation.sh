#!/bin/bash

CONFIG=$1

echo "================================="
echo " Generating Input: $CONFIG"
echo "================================="

conda run -n que python input_generation.py --config "configs/$CONFIG"
echo "done"

