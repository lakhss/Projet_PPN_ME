#pragma once

#include "DecisionTreeRegressor.hpp"
#include "Matrix.hpp"

#include <vector>
#include <random>

class BaggingRegressor {
public:
    int n_estimators = 15;
    double sample_ratio = 1.0;
    int max_depth = 10;
    int min_samples_split = 6;
    unsigned int seed = 42;

    void fit(const Matrix& X,
             const std::vector<double>& y);

    double predict(const std::vector<double>& x) const;

private:
    std::vector<DecisionTreeRegressor> trees;
};