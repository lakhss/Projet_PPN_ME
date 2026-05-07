#include "HistogramBoostingRegressor.hpp"

#include <numeric>
#include <iostream>
#include <omp.h>

void HistogramBoostingRegressor::fit(const QuantizedDataset& dataset) {
    trees.clear();
    trees.reserve(n_estimators);
    
    const std::size_t N = dataset.get_num_samples();
    const auto& y = dataset.get_targets();

    if (N == 0 || y.empty()) {
        std::cerr << "HistogramBoosting: dataset vide ou invalide." << std::endl;
        init_value = 0.0;
        return;
    }

    init_value = std::accumulate(y.begin(), y.end(), 0.0) / y.size();

    std::vector<double> preds(N, init_value);
    std::vector<double> residuals(N);

    std::cout << "[OpenMP] Lancement du Boosting HPC sur "
              << omp_get_max_threads()
              << " threads..." << std::endl;


    QuantizedDataset working_dataset = dataset;

    for (int m = 0; m < n_estimators; ++m) {

        // 1. Calcul des residus en parallele
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < static_cast<int>(N); ++i) {
            residuals[i] = y[i] - preds[i];
        }

        working_dataset.set_targets(residuals);


        // 3. Construction de l'arbre histogramme
        // La boucle sur les estimateurs reste sequentielle,
        // mais l'arbre lui-même peut etre parallele si HistogramTreeRegressor l'est.
        trees.emplace_back();
        trees[m].max_depth = max_depth;
        trees[m].min_samples_split = min_samples_split;
        trees[m].n_bins = n_bins;


        trees.back().fit(working_dataset);

        // 4. Mise a jour des predictions en parallele
        // #pragma omp parallel for schedule(static)
        for (int i = 0; i < static_cast<int>(N); ++i) {
            preds[i] += learning_rate * trees[m].predict(dataset.get_raw_row(i));
        }
    }
}

double HistogramBoostingRegressor::predict(const std::vector<double>& x) const {
    double sum = 0.0;

    // #pragma omp parallel for reduction(+:sum) schedule(static)
    for (int m = 0; m < static_cast<int>(trees.size()); ++m) {
        sum += learning_rate * trees[m].predict(x);
    }

    return init_value + sum;
}