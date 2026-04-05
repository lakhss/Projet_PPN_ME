#include "BoostingRegressor.hpp"
#include "Matrix.hpp"

#include <numeric>
#include <iostream>

/**
 * @brief Train the Boosting Regressor on the given data
 */
void BoostingRegressor::fit(const Matrix& X,
                            const std::vector<double>& y) {

    trees.clear();
    trees.reserve(n_estimators);

    if (X.empty() || y.empty() || X.rows() != y.size()) {
        std::cerr << "Boosting: dataset vide ou invalide." << std::endl;
        init_value = 0.0;
        return;
    }

    // Initial prediction = mean of y
    init_value = std::accumulate(y.begin(), y.end(), 0.0) / y.size();

    std::vector<double> preds(y.size(), init_value);
    std::vector<double> residuals(y.size());

    for (int m = 0; m < n_estimators; ++m) {

        // compute residuals
        for (size_t i = 0; i < y.size(); ++i) {
            residuals[i] = y[i] - preds[i];
        }

        // train tree
        DecisionTreeRegressor tree;
        tree.max_depth = max_depth;
        tree.min_samples_split = min_samples_split;
        tree.fit(X, residuals);

        // update predictions
        for (size_t i = 0; i < X.rows(); ++i) {
            std::vector<double> row = X.row(i); // conversion simple
            preds[i] += learning_rate * tree.predict(row);
        }

        trees.push_back(std::move(tree));
    }
}

/**
 * @brief Predict value
 */
double BoostingRegressor::predict(const std::vector<double>& x) const {
    double y_pred = init_value;

    for (const auto& t : trees) {
        y_pred += learning_rate * t.predict(x);
    }

    return y_pred;
}