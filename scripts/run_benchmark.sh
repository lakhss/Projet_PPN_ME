#!/bin/bash

set -e

OUTPUT_FILE="benchmark_results.csv"
PLOT_FILE="benchmark_plot.png"

echo "Running C++ benchmark..."

../build/main > raw_output.txt

echo "Parsing results into CSV..."

python3 parse_results_boost.py raw_output.txt $OUTPUT_FILE

echo "Generating plot..."

python3 double_plot_boost.py $OUTPUT_FILE $PLOT_FILE

echo "Done."
echo "CSV saved to: $OUTPUT_FILE"
echo "Plot saved to: $PLOT_FILE"