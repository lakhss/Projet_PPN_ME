#!/bin/bash

EXEC="../build/main"   # <-- change ça
OUT_CSV="results_raw.csv"
FINAL_CSV="results_mean.csv"

BINS=(2 4 8 16 32 64 128 256 512 1024)
RUNS=10

echo "run,bin,rmse,mape,time" > $OUT_CSV

# ====== RUN BENCHMARK ======
for r in $(seq 1 $RUNS); do
  echo "Run $r/$RUNS"

  OUTPUT=$($EXEC)

  for b in "${BINS[@]}"; do
    LINE=$(echo "$OUTPUT" | grep "HistTree (D=12, B=$b)")

    RMSE=$(echo $LINE | awk '{print $3}')
    MAPE=$(echo $LINE | awk '{print $4}')
    TIME=$(echo $LINE | awk '{print $5}')

    echo "$r,$b,$RMSE,$MAPE,$TIME" >> $OUT_CSV
  done
done

# ====== COMPUTE MEAN ======
echo "bin,rmse_mean,time_mean" > $FINAL_CSV

for b in "${BINS[@]}"; do
  rmse_avg=$(awk -F',' -v b=$b '$2==b {sum+=$3; n++} END {print sum/n}' $OUT_CSV)
  time_avg=$(awk -F',' -v b=$b '$2==b {sum+=$5; n++} END {print sum/n}' $OUT_CSV)

  echo "$b,$rmse_avg,$time_avg" >> $FINAL_CSV
done

echo "Done -> $FINAL_CSV"

python3 plot.py $FINAL_CSV