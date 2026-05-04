#include "HistogramBoostingRegressor.hpp"

#include <numeric>
#include <iostream>

void HistogramBoostingRegressor::fit(const Matrix& X,
                                     const std::vector<double>& y) {
    trees.clear();
    trees.reserve(n_estimators);

    if (X.empty() || y.empty() || X.rows() != y.size()) {
        std::cerr << "HistogramBoosting: dataset vide ou invalide." << std::endl;
        init_value = 0.0;
        return;
    }

    init_value = std::accumulate(y.begin(), y.end(), 0.0) / y.size();

    std::vector<double> preds(y.size(), init_value);
    std::vector<double> residuals(y.size());
    #pragma omp parallel for
    for (int m = 0; m < n_estimators; ++m) {
        for (size_t i = 0; i < y.size(); ++i) {
            residuals[i] = y[i] - preds[i];
        }

        HistogramTreeRegressor tree;
        tree.max_depth = max_depth;
        tree.min_samples_split = min_samples_split;
        tree.n_bins = n_bins;
        tree.fit(X, residuals);
      
        for (size_t i = 0; i < X.rows(); ++i) {
            preds[i] += learning_rate * tree.predict(X.row(i));
        }

        trees.push_back(std::move(tree));
    }
}

double HistogramBoostingRegressor::predict(const std::vector<double>& x) const {
    double y_pred = init_value;
    for (const auto& t : trees) {
        y_pred += learning_rate * t.predict(x);
    }
    return y_pred;
}