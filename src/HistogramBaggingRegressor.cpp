#include "HistogramBaggingRegressor.hpp"

#include <algorithm>
#include <iostream>
#include <random>

void HistogramBaggingRegressor::fit(const Matrix& X,
                                    const std::vector<double>& y) {
    trees.clear();
    trees.reserve(n_estimators);

    const size_t N = X.rows();
    if (N == 0 || y.empty() || X.rows() != y.size()) {
        std::cerr << "HistogramBagging: dataset vide ou invalide." << std::endl;
        return;
    }

    const size_t m = std::max<size_t>(1, static_cast<size_t>(sample_ratio * N));

    std::mt19937 gen(seed);
    std::uniform_int_distribution<size_t> dist(0, N - 1);

    for (int b = 0; b < n_estimators; ++b) {
        Matrix Xb(m, X.cols());
        std::vector<double> yb;
        yb.reserve(m);

        for (size_t i = 0; i < m; ++i) {
            size_t j = dist(gen);
            for (size_t c = 0; c < X.cols(); ++c) {
                Xb(i, c) = X(j, c);
            }
            yb.push_back(y[j]);
        }

        trees.emplace_back();
        trees.back().max_depth = max_depth;
        trees.back().min_samples_split = min_samples_split;
        trees.back().n_bins = n_bins;
        trees.back().fit(Xb, yb);
    }
}

double HistogramBaggingRegressor::predict(const std::vector<double>& x) const {
    if (trees.empty()) return 0.0;

    double sum = 0.0;
    for (const auto& t : trees) {
        sum += t.predict(x);
    }
    return sum / trees.size();
}