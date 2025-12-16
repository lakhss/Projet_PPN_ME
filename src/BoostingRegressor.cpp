#include "BoostingRegressor.hpp"
#include <numeric>

void BoostingRegressor::fit(const std::vector<std::vector<double>>& X,
                            const std::vector<double>& y) {

    trees.clear();
    trees.reserve(n_estimators);

    // Initialisation : prédiction constante = moyenne de y
    init_value = std::accumulate(y.begin(), y.end(), 0.0) / y.size();

    std::vector<double> preds(y.size(), init_value);
    std::vector<double> residuals(y.size());

    for (int m = 0; m < n_estimators; ++m) {
        // Calcul des résidus
        for (size_t i = 0; i < y.size(); ++i)
            residuals[i] = y[i] - preds[i];

        // Entraîner un arbre sur les résidus
        DecisionTreeRegressor tree;
        tree.max_depth = max_depth;
        tree.min_samples_split = min_samples_split;
        tree.fit(X, residuals);

        // Mise à jour des prédictions
        for (size_t i = 0; i < X.size(); ++i)
            preds[i] += learning_rate * tree.predict(X[i]);

        trees.push_back(std::move(tree));
    }
}

double BoostingRegressor::predict(const std::vector<double>& x) const {
    double y_pred = init_value;
    for (const auto& t : trees)
        y_pred += learning_rate * t.predict(x);
    return y_pred;
}