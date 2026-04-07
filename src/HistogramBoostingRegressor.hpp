#pragma once

#include "HistogramTreeRegressor.hpp"
#include "Matrix.hpp"

#include <vector>

class HistogramBoostingRegressor {
public:
    int n_estimators = 50;
    double learning_rate = 0.1;
    int max_depth = 3;
    int min_samples_split = 6;
    int n_bins = 16;

    void fit(const Matrix& X, const std::vector<double>& y);
    double predict(const std::vector<double>& x) const;

private:
    double init_value = 0.0;
    std::vector<HistogramTreeRegressor> trees;
};