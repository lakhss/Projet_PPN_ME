import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

input_file = "scripts/final_comparison.csv"
data = []

if not os.path.exists(input_file):
    print(f"Erreur : {input_file} introuvable. Lance le benchmark d'abord.")
    exit(1)

with open(input_file, 'r') as f:
    for line in f:
        parts = line.split()
        # On ne garde que les lignes qui commencent par Bagging ou Boosting
        # et qui ont assez de colonnes (au moins 6)
        if len(parts) >= 6 and (parts[0] == "Bagging" or parts[0] == "Boosting"):
            try:
                data.append({
                    "Algorithme": parts[0],
                    "Version": parts[1],
                    "N": parts[2].replace("(N=", "").replace(")", ""),
                    "RMSE": float(parts[3]),
                    "Temps": float(parts[5])
                })
            except ValueError:
                continue # Ignore les lignes mal formatées

df = pd.DataFrame(data)

if df.empty:
    print("Aucune donnée valide trouvée dans le CSV. Vérifie le format de sortie.")
    exit(1)

# Création du graphique
plt.figure(figsize=(12, 7))
sns.set_theme(style="whitegrid")

# On utilise l'échelle Log car le gap entre 38s et 0.2s est trop grand
ax = sns.barplot(x="Algorithme", y="Temps", hue="Version", data=df[df['N'] == '100'])
ax.set_yscale("log") 

plt.title("Performance N=100 : Naïf (S1) et Parallèle (S2)", fontsize=14)
plt.ylabel("Temps d'exécution (secondes)", fontsize=12)

# Ajout des étiquettes de temps sur les barres
for p in ax.patches:
    if p.get_height() > 0:
        ax.annotate(f'{p.get_height():.3f}s', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 10), 
                    textcoords='offset points', fontsize=10, fontweight='bold')

plt.savefig("scripts/speedup_hpc.png")
print("Succès ! Graphique généré : scripts/speedup_hpc.png")