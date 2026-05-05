import sys
import pandas as pd
import matplotlib.pyplot as plt

csv_file = sys.argv[1]
output_image = sys.argv[2]

df = pd.read_csv(csv_file)

naive = df[df["model"] == "Arbre Naïf"]
hist = df[df["model"] == "HistTree HPC"]

# Sort by depth
naive = naive.sort_values("depth")
hist = hist.sort_values("depth")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ---- TIME ----
axes[0].plot(naive["depth"], naive["time"], marker="o", label="Naïf")
axes[0].plot(hist["depth"], hist["time"], marker="o", label="Hist (HPC)")
axes[0].set_title("Temps d'exécution")
axes[0].set_xlabel("Profondeur")
axes[0].set_ylabel("Temps (s)")
axes[0].legend()
axes[0].grid(True)

# ---- ERROR (RMSE) ----
axes[1].plot(naive["depth"], naive["rmse"], marker="o", label="Naïf RMSE")
axes[1].plot(hist["depth"], hist["rmse"], marker="o", label="Hist RMSE")
axes[1].set_title("Erreur (RMSE)")
axes[1].set_xlabel("Profondeur")
axes[1].set_ylabel("RMSE")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig(output_image, dpi=200)

print(f"[OK] Plot saved to {output_image}")