import sys
import re
import csv

input_file = sys.argv[1]
output_file = sys.argv[2]

pattern = re.compile(
    r"^(Arbre Naïf|HistTree HPC)\s*\(D=(\d+)\)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
)

rows = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            model = match.group(1)
            depth = int(match.group(2))
            rmse = float(match.group(3))
            mape = float(match.group(4))
            time = float(match.group(5))

            rows.append([model, depth, rmse, mape, time])

with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["model", "depth", "rmse", "mape", "time"])
    writer.writerows(rows)

print(f"[OK] CSV written to {output_file}")