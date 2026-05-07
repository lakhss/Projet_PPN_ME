#!/bin/bash

set -e

OUTPUT_FILE="../csv/parsed_results.csv"
PLOT_FILE="../png/parsed_results.png"

# ---- set number of OpenMP threads ----
## export OMP_NUM_THREADS=8  

# echo "Running C++ benchmark with $OMP_NUM_THREADS threads..."

../../build/main > ../raw_output.txt

echo "Parsing results into CSV..."

python3 ../python/parse_results_boost.py $OUTPUT_FILE

echo "Generating plot..."

python3 ../python/plot_results_boost.py $OUTPUT_FILE $PLOT_FILE

echo "Done."
echo "CSV saved to: $OUTPUT_FILE"
echo "Plot saved to: $PLOT_FILE"