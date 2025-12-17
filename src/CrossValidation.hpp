#pragma once
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>

class CrossValidation {
public:
    // Structure simple pour stocker les indices d'un fold
    struct FoldIndices {
        std::vector<size_t> train_idx;
        std::vector<size_t> test_idx;
    };

    /**
     * Génère k paires d'indices (Train, Test).
     * @param n_samples : Nombre total de lignes dans le dataset
     * @param k : Nombre de folds (ex: 5 ou 10)
     * @param seed : Graine aléatoire pour le mélange
     */
    static std::vector<FoldIndices> k_fold_split(size_t n_samples, int k, int seed = 42) {
        std::vector<size_t> indices(n_samples);
        std::iota(indices.begin(), indices.end(), 0);
        
        // Mélange aléatoire des indices
        std::mt19937 g(seed);
        std::shuffle(indices.begin(), indices.end(), g);

        std::vector<FoldIndices> folds;
        folds.reserve(k);

        size_t fold_size = n_samples / k;
        
        for (int i = 0; i < k; ++i) {
            size_t start = i * fold_size;
            size_t end = (i == k - 1) ? n_samples : (i + 1) * fold_size;

            FoldIndices fold;
            fold.train_idx.reserve(n_samples - (end - start));
            fold.test_idx.reserve(end - start);

            // On parcourt tous les indices mélangés
            for (size_t j = 0; j < n_samples; ++j) {
                // if indice est dans la fenêtre actuelle [start, end[, c'est du TEST
                if (j >= start && j < end) {
                    fold.test_idx.push_back(indices[j]);
                } else {
                    // Sinon c'est du TRAIN
                    fold.train_idx.push_back(indices[j]);
                }
            }
            folds.push_back(fold);
        }
        return folds;
    }
};