import sys
import pandas as pd
import matplotlib.pyplot as plt

csv_file = sys.argv[1]
output_image = sys.argv[2]

df = pd.read_csv(csv_file)

naive = df[df["model"] == "Arbre Naïf"].sort_values("depth")
hist = df[df["model"] == "HistTree HPC"].sort_values("depth")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

# -----------------------
# TOP: TIME
# -----------------------
ax1.plot(naive["depth"], naive["time"], marker="o", label="Naïf")
ax1.plot(hist["depth"], hist["time"], marker="o", label="HistTree HPC")

ax1.set_ylabel("Temps (s)")
ax1.set_title("Performance : Temps d'exécution vs Profondeur")
ax1.grid(True)
ax1.legend()

# -----------------------
# BOTTOM: ERROR
# -----------------------
ax2.plot(naive["depth"], naive["rmse"], marker="s", linestyle="--", label="Naïf RMSE")
ax2.plot(hist["depth"], hist["rmse"], marker="s", linestyle="--", label="HistTree RMSE")

ax2.set_xlabel("Profondeur")
ax2.set_ylabel("RMSE")
ax2.set_title("Précision : Erreur vs Profondeur")
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.savefig(output_image, dpi=200)

print(f"[OK] Saved plot to {output_image}")