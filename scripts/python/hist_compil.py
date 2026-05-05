import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("compil.csv", header=None, names=["flags", "time"])

df["time"] = pd.to_numeric(df["time"], errors="coerce")

df = df.dropna()

baseline = df["time"].iloc[0]

df["speedup"] = baseline / df["time"]

plt.figure()

bars = plt.bar(df["flags"], df["speedup"])

plt.xlabel("Flags de compilation")
plt.ylabel("Speedup (vs -O0)")
plt.title("Impact des optimisations")

plt.xticks(rotation=45)
plt.grid(axis="y")

for bar, val in zip(bars, df["speedup"]):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height(),
             f"{val:.2f}",
             ha="center",
             va="bottom")

plt.tight_layout()
plt.savefig("benchmark_speedup.png", dpi=300)
plt.show()