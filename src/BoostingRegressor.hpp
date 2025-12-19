#pragma once
#include "DecisionTreeRegressor.hpp"
#include <vector>

class BoostingRegressor {
public:
    int n_estimators = 50;
    double learning_rate = 0.1;
    int max_depth = 3;
    int min_samples_split = 6;

    void fit(const std::vector<std::vector<double>>& X,
             const std::vector<double>& y);

    double predict(const std::vector<double>& x) const;

private:
    double init_value = 0.0;
    std::vector<DecisionTreeRegressor> trees;
};