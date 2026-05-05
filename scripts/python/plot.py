import pandas as pd
import matplotlib.pyplot as plt
import sys

file = sys.argv[1]

df = pd.read_csv(file)

bins = df["bin"]

# ===== RMSE =====
plt.figure()
plt.plot(bins, df["rmse_mean"], marker="o")
plt.xscale("log")
plt.xlabel("Number of bins")
plt.ylabel("RMSE")
plt.title("Impact du nombre de bins sur RMSE")
plt.grid(True)
plt.show()

# ===== TIME =====
plt.figure()
plt.plot(bins, df["time_mean"], marker="o")
plt.xscale("log")
plt.xlabel("Number of bins")
plt.ylabel("Time (s)")
plt.title("Impact du nombre de bins sur le temps d'exécution")
plt.grid(True)
plt.show()