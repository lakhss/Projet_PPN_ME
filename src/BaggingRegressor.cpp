#include "BaggingRegressor.hpp"
#include <algorithm>
#include <iostream>
#include <random>

/**
 * @brief Train the Bagging regression model.
 *
 * This method implements the Bagging (Bootstrap Aggregating) principle:
 * - Multiple regression trees are trained
 * - Each tree is trained on a bootstrap sample of the dataset
 * - Samples are drawn uniformly with replacement
 *
 * @param X Feature matrix (N samples × d features)
 * @param y Target vector associated with the samples
 */

void BaggingRegressor::fit(const std::vector<std::vector<double>>& X,
                           const std::vector<double>& y) {
    trees.clear();
    trees.reserve(n_estimators);

    const size_t N = X.size();

    /**
     * @brief Size of the bootstrap sample
     *
     * Defined as a fraction of the total dataset size,
     * with a minimum value of 1.
     */
    const size_t m = std::max<size_t>(1, (size_t)(sample_ratio * N));

    std::mt19937 gen(seed);
    std::uniform_int_distribution<size_t> dist(0, N - 1);

    /**
     * @brief Main bagging loop
     *
     * At each iteration:
     * - A bootstrap sample is generated
     * - A regression tree is trained on this sample
     */

    for (int b = 0; b < n_estimators; ++b) {
        std::vector<std::vector<double>> Xb;
        std::vector<double> yb;
        Xb.reserve(m);
        yb.reserve(m);

        // Bootstrap uniforme (tirage avec remise)
        for (size_t i = 0; i < m; ++i) {
            size_t j = dist(gen);
            Xb.push_back(X[j]);
            yb.push_back(y[j]);
        }

        trees.emplace_back();
        trees.back().max_depth = max_depth;
        trees.back().min_samples_split = min_samples_split;
        trees.back().fit(Xb, yb);
    }

    std::cout << "Bagging terminé (" << n_estimators
              << " arbres, max_depth = " << max_depth
              << ", min_samples_split = " << min_samples_split
              << ")\n";
}


/**
 * @brief Predict a target value for a given observation.
 *
 * The final prediction is computed as the arithmetic mean
 * of the predictions produced by all trained trees.
 *
 * @param x Feature vector of the observation to predict
 * @return Predicted value produced by the bagging model
 */

double BaggingRegressor::predict(const std::vector<double>& x) const {
    if (trees.empty()) return 0.0;

    double sum = 0.0;
    for (const auto& t : trees) {
        sum += t.predict(x);
    }
    return sum / trees.size();
}