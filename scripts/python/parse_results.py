import csv
import re

input_file = "../raw_output.txt"
output_file = "../csv/parsed_results.csv"

with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

rows = []
start = False

for line in lines:
    line = line.strip()

    if line.startswith("Modele"):
        start = True
        continue

    if start:
        if line == "" or line.startswith("Terminé"):
            break

        parts = re.split(r"\s{2,}", line)

        if len(parts) >= 4:
            try:
                match = re.search(r"D=(\d+)", parts[0])
                depth = int(match.group(1)) if match else None

                model = parts[0].split(" (")[0]

                rows.append([
                    model,
                    depth,
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3])
                ])
            except:
                pass

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["model", "depth", "rmse", "mape", "time"])
    writer.writerows(rows)

print("Done:", output_file)