#!/bin/bash

input_file="$1"
output_file="expected_lhood.csv"

echo "macroreplication,bayesopt_iteration,inference_iteration,expected_lhood" > "$output_file"

awk '
BEGIN {
    macro_iter = 0
    bayes_iter = 0
    inf_iter = 0
}

# Detect macroreplication start
/Starting macroreplication/ {
    macro_iter = $3
    bayes_iter = 0
    inf_iter = 0
}

# Detect BayesOpt iteration start
/Starting iter/ {
    bayes_iter = $3
    inf_iter = 0
}

# Detect Expected Lhood values
/Expected Lhood:/ {
    inf_iter += 1

    split($0, parts, ":")
    gsub(/ /, "", parts[2])

    print macro_iter "," bayes_iter "," inf_iter "," parts[2]
}
' "$input_file" >> "$output_file"

echo "Saved output to $output_file"
