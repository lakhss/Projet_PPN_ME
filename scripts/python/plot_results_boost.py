import sys
import pandas as pd
import matplotlib.pyplot as plt

csv_file = sys.argv[1]
output_image = sys.argv[2]

df = pd.read_csv(csv_file)

naive = df[df["model"] == "Boosting Naïf"]
hist = df[df["model"] == "Boosting HPC"]

naive = naive.sort_values("n_estimators")
hist = hist.sort_values("n_estimators")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ---- TIME ----
axes[0].plot(naive["n_estimators"], naive["time"], marker="o",color="blue", label="Naïf")
axes[0].plot(hist["n_estimators"], hist["time"], marker="o",color="red", label="Hist/SoA")
axes[0].set_title("Execution time", fontsize=18)
axes[0].set_xlabel("Trees", fontsize=14)
axes[0].set_ylabel("Time (s)", fontsize=14)
axes[0].legend()
axes[0].grid(True)

# ---- ERROR (RMSE) ----
axes[1].plot(naive["n_estimators"], naive["rmse"], marker="o",color="blue", label="Naïf RMSE")
axes[1].plot(hist["n_estimators"], hist["rmse"], marker="o",color="red", label="Hist/SoA RMSE")
axes[1].set_title("Error (RMSE)", fontsize=18)
axes[1].set_xlabel("Trees", fontsize=14)
axes[1].set_ylabel("RMSE", fontsize=14)
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig(output_image, dpi=200)

print(f"[OK] Plot saved to {output_image}")