import subprocess
import re

runs = 10

rmse_list = []
mape_list = []
time_list = []

for i in range(runs):
    print(f"Run {i+1}/{runs}...")

    result = subprocess.run(["../build/main"], capture_output=True, text=True)
    output = result.stdout

    # Match: RMSE MAPE TIME (3 floats after model name)
    match = re.search(
        r"Arbre Naïf.*?(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)",
        output
    )

    if match:
        rmse = float(match.group(1))
        mape = float(match.group(2))
        t = float(match.group(3))

        rmse_list.append(rmse)
        mape_list.append(mape)
        time_list.append(t)

        print(f"  RMSE={rmse}, MAPE={mape}, Time={t}s")
    else:
        print("  ⚠️ Failed to parse output")

# Averages
avg_rmse = sum(rmse_list) / len(rmse_list)
avg_mape = sum(mape_list) / len(mape_list)
avg_time = sum(time_list) / len(time_list)

print("\n==============================")
print("AVERAGE OVER 10 RUNS")
print("==============================")
print(f"RMSE: {avg_rmse:.6f}")
print(f"MAPE: {avg_mape:.6f}")
print(f"TIME: {avg_time:.6f} s")
print("==============================")