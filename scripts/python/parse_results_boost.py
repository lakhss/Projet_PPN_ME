import sys
import re
import csv

input_file = "../raw_output.txt"
output_file = sys.argv[1]

# Match:
# Boosting Naïf (N=10)         0.057252    47.358954   4.015582
# Boosting HPC (N=30)          0.044857    29.786906   0.335426

pattern = re.compile(
    r"^(Boosting Naïf|Boosting HPC)\s*\(N=(\d+)\)\s+"
    r"([\d.]+)\s+([\d.]+)\s+([\d.]+)"
)

rows = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        # skip noise lines
        if "OpenMP" in line:
            continue

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