import sys
import pandas as pd
import matplotlib.pyplot as plt

csv_file = sys.argv[1]
output_image = sys.argv[2]

df = pd.read_csv(csv_file)

naive = df[df["model"] == "Arbre Naïf"].sort_values("depth")
hist = df[df["model"] == "HistTree HPC"].sort_values("depth")

fig, ax1 = plt.subplots(figsize=(10, 5))

# ---- TIME (left axis) ----
ax1.set_xlabel("Profondeur")
ax1.set_ylabel("Temps (s)")

ax1.plot(
    naive["depth"], naive["time"],
    marker="o",
    color="blue",
    label="Naïf - Temps"
)

ax1.plot(
    hist["depth"], hist["time"],
    marker="o",
    color="red",
    label="Hist - Temps"
)

ax1.tick_params(axis='y')
ax1.grid(True)

# ---- ERROR (right axis) ----
ax2 = ax1.twinx()
ax2.set_ylabel("RMSE")

ax2.plot(
    naive["depth"], naive["rmse"],
    linestyle="--",
    marker="s",
    color="blue",
    label="Naïf - RMSE"
)

ax2.plot(
    hist["depth"], hist["rmse"],
    linestyle="--",
    marker="s",
    color="red",
    label="Hist - RMSE"
)

ax2.tick_params(axis='y')

# ---- LEGEND (combined, middle right) ----
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()

ax1.legend(
    lines_1 + lines_2,
    labels_1 + labels_2,
    loc="center right"
)

plt.title("Performance vs Précision : Naïf vs HistTree HPC")
plt.tight_layout()
plt.savefig(output_image, dpi=200)

print(f"[OK] Saved to {output_image}")