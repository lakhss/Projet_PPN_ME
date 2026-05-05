import sys
import pandas as pd
import matplotlib.pyplot as plt

csv_file = sys.argv[1]
output_image = sys.argv[2]

df = pd.read_csv(csv_file)

naive = df[df["model"] == "Bagging Naïf"].sort_values("n_estimators")
hist = df[df["model"] == "Bagging HPC"].sort_values("n_estimators")

fig, ax1 = plt.subplots(figsize=(10, 5))

# ---- TIME (left axis) ----
ax1.set_xlabel("Number of estimators (N)", fontsize=16)
ax1.set_ylabel("Time (s) [logarithmic scale]", fontsize=16)

ax1.set_yscale("log")

ax1.plot(
    naive["n_estimators"], naive["time"],
    marker="o",
    color="blue",
    label="Naïf - Temps"
)

ax1.plot(
    hist["n_estimators"], hist["time"],
    marker="o",
    color="red",
    label="HPC - Temps"
)

ax1.tick_params(axis='both', labelsize=12)
ax1.grid(True)

# ---- ERROR (right axis) ----
ax2 = ax1.twinx()
ax2.set_ylabel("RMSE", fontsize=16)

ax2.plot(
    naive["n_estimators"], naive["rmse"],
    linestyle="--",
    marker="s",
    color="blue",
    label="Naïf - RMSE"
)

ax2.plot(
    hist["n_estimators"], hist["rmse"],
    linestyle="--",
    marker="s",
    color="red",
    label="HPC - RMSE"
)

ax2.tick_params(axis='y', labelsize=12)

# ---- LEGEND ----
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()

ax1.legend(
    lines_1 + lines_2,
    labels_1 + labels_2,
    loc="center right",
    bbox_to_anchor=(1, 0.75),
    fontsize=12
)

plt.title("Bagging : Performance vs Precision (Naive vs Histogram-based split)", fontsize=18)

plt.tight_layout()
plt.savefig(output_image, dpi=200)

print(f"[OK] Saved to {output_image}")