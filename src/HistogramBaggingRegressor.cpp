#include "HistogramBaggingRegressor.hpp"

#include <iostream>
#include <random>
#include <omp.h> 

void HistogramBaggingRegressor::fit(const QuantizedDataset& dataset) {
    trees.clear();
    
    // std::vector::emplace_back a ne pas mettre Jam n'a pas aimer et il a raison 
    trees.resize(n_estimators); 

    const std::size_t N = dataset.get_num_samples();
    if (N == 0) {
        std::cerr << "HistogramBagging: dataset vide ou invalide." << std::endl;
        return;
    }

    const std::size_t m = std::max<std::size_t>(1, static_cast<std::size_t>(sample_ratio * N));

    std::cout << "[OpenMP] Lancement du Bagging HPC sur " << omp_get_max_threads() << " threads..." << std::endl;


    //#pragma omp parallel for schedule(dynamic)
    for (int b = 0; b < n_estimators; ++b) {
        
        std::mt19937 local_gen(seed + b * 19937); 
        std::uniform_int_distribution<std::size_t> dist(0, N - 1);

        // On ne copie aucune donnée on génère juste 'm' indices aléatoires. BOOTSTRAP
        std::vector<std::size_t> bootstrap_indices(m);
        for (std::size_t i = 0; i < m; ++i) {
            bootstrap_indices[i] = dist(local_gen);
        }

        trees[b].max_depth = max_depth;
        trees[b].min_samples_split = min_samples_split;
        trees[b].n_bins = n_bins;
        
        trees[b].fit_bootstrap(dataset, bootstrap_indices); 
    }
}

double HistogramBaggingRegressor::predict(const std::vector<double>& x) const {
    if (trees.empty()) return 0.0;

    double sum = 0.0;
    
    // Le processeur additionne en cache L1 local, puis fait la somme finale
    
    //#pragma omp parallel for reduction(+:sum)
    for (int b = 0; b < n_estimators; ++b) {
        sum += trees[b].predict(x);
    }
    
    return sum / static_cast<double>(trees.size());
}