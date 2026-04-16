#!/bin/bash
echo "Reading in from experiments.txt"
while read -r file; do

	./run_experiment.sh "$file"

done < experiments.txt
