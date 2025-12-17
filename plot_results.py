import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)

RESULTS_FILE = "build/results.csv"

if not os.path.exists(RESULTS_FILE):
    sys.exit(f"Erreur : {RESULTS_FILE} introuvable.")

try:
    df = pd.read_csv(RESULTS_FILE)
except Exception as e:
    sys.exit(f"Erreur lecture CSV : {e}")

print(f"Données chargées : {len(df)} lignes trouvées.")

# ===========================================================
# 1. GÉNÉRATION DES GRAPHES PAR DATASET
# ===========================================================
def generate_per_dataset_graphs(ds_name, data):
    print(f"-> Traitement : {ds_name}")
    
    # A. Profondeur Arbre
    depth_data = data[data['Model'].str.contains(r"Depth=")].copy()
    if not depth_data.empty:
        depth_data['Depth'] = depth_data['Model'].str.extract(r'Depth=(\d+)').astype(int)
        
        fig, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.set_xlabel('Profondeur')
        ax1.set_ylabel('RMSE', color=color, fontweight='bold')
        ax1.plot(depth_data['Depth'], depth_data['RMSE'], color=color, marker='o')
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Temps (s)', color=color)
        ax2.plot(depth_data['Depth'], depth_data['Time'], color=color, marker='x', linestyle='--')
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title(f"[{ds_name}] Arbre Unique : Profondeur")
        plt.tight_layout()
        plt.savefig(f"{ds_name}_1_profondeur.png")
        plt.close()

    # B. Duel Bagging vs Boosting
    ensemble_data = data[data['Model'].str.contains(r"\(N=")].copy()
    if not ensemble_data.empty:
        ensemble_data['N_Trees'] = ensemble_data['Model'].str.extract(r'N=(\d+)').astype(int)
        ensemble_data['Technique'] = ensemble_data['Model'].apply(
            lambda x: 'Boosting' if 'Boosting' in x else 'Bagging'
        )
        
        plt.figure()
        sns.lineplot(data=ensemble_data, x="N_Trees", y="RMSE", hue="Technique", style="Technique", markers=True, linewidth=2.5)
        plt.title(f"[{ds_name}] Bagging vs Boosting")
        plt.ylabel("RMSE")
        plt.xlabel("Nombre d'Arbres")
        plt.tight_layout()
        plt.savefig(f"{ds_name}_2_duel.png")
        plt.close()

unique_datasets = df['Dataset'].unique()
for ds in unique_datasets:
    ds_data = df[df['Dataset'] == ds]
    generate_per_dataset_graphs(ds, ds_data)

# ===========================================================
# 2. GRAPHIQUE FINAL : COMPARAISON INTER-DATASETS
# ===========================================================
print("-> Génération du résumé global...")

# On veut comparer les MEILLEURES versions de chaque algo pour chaque dataset
# On va simplifier les noms pour le groupement
def simplify_name(model_name):
    if "Arbre" in model_name: return "Arbre Unique"
    if "Bagging" in model_name: return "Bagging"
    if "Boosting" in model_name: return "Boosting"
    return "Autre"

df['Algo_Type'] = df['Model'].apply(simplify_name)

# Pour chaque Dataset et chaque Algo_Type, on prend le meilleur RMSE (le minimum)
best_results = df.groupby(['Dataset', 'Algo_Type'])['RMSE'].min().reset_index()

plt.figure(figsize=(12, 6))
sns.barplot(
    data=best_results, 
    x="Dataset", 
    y="RMSE", 
    hue="Algo_Type", 
    palette="viridis",
    edgecolor=".2"
)

plt.title("Comparaison Globale : Qui est le meilleur sur chaque Dataset ?", fontsize=15)
plt.ylabel("Meilleur RMSE atteint (Plus bas = Mieux)")
plt.xlabel("Jeu de Données")
plt.legend(title="Algorithme", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig("GLOBAL_COMPARAISON.png")
print("Terminé  Image : GLOBAL_COMPARAISON.png")