#!/bin/bash

# On se place à la racine du projet
cd "$(dirname "$0")/.."

# 1. Préparation du dossier de build
mkdir -p build
cd build

# 2. Configuration et Compilation via CMake
echo "--- Configuration CMake (Release) ---"
cmake .. -DCMAKE_BUILD_TYPE=Release

echo "--- Compilation ---"
make -j$(nproc)

# Vérification de la réussite
if [ ! -f "./main" ]; then
    echo "Erreur : L'exécutable 'main' n'a pas été généré."
    exit 1
fi

# 3. Préparation du fichier de résultats
# On remonte d'un cran pour que le chemin scripts/ soit correct
cd ..
OUTPUT="scripts/final_comparison.csv"
echo "Dataset;Model;RMSE;MAE;MAPE;Time" > $OUTPUT

# 4. Lancement du Benchmark (Mode 8)
echo "Lancement des tests (HPC & Naïf)..."

# Fixation des threads pour la partie HPC
export OMP_NUM_THREADS=8

# On lance l'exécutable depuis le dossier build
# On redirige la sortie pour capturer les résultats
./build/main 8 >> $OUTPUT

echo "--- Terminé ! Résultats disponibles dans $OUTPUT ---"