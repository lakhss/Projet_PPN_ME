import sys
import re
import csv

input_file = sys.argv[1]
output_file = sys.argv[2]

# Match:
# Bagging Naïf (N=10)   0.036280   14.85   9.49
pattern = re.compile(
    r"^(Bagging Naïf|Bagging HPC)\s*\(N=(\d+)\)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
)

rows = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        match = pattern.search(line)
        if match:
            model = match.group(1)
            n_estimators = int(match.group(2))
            rmse = float(match.group(3))
            mape = float(match.group(4))
            time = float(match.group(5))

            rows.append([model, n_estimators, rmse, mape, time])

with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["model", "n_estimators", "rmse", "mape", "time"])
    writer.writerows(rows)

print(f"[OK] CSV written to {output_file}")