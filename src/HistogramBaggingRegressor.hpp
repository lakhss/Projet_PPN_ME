#pragma once

#include "HistogramTreeRegressor.hpp"
#include "QuantizedDataset.hpp"

#include <vector>
#include <random>

class HistogramBaggingRegressor {
public:
    int n_estimators = 15;
    double sample_ratio = 1.0;
    int max_depth = 10;
    std::size_t min_samples_split = 6; 
    int n_bins = 256; // a 256 pour saturer les uint8_t mme fonctionnement que sur quantized (voir)
    unsigned int seed = 42;

    HistogramBaggingRegressor() = default;
    ~HistogramBaggingRegressor() = default;

    HistogramBaggingRegressor(const HistogramBaggingRegressor&) = delete;
    HistogramBaggingRegressor& operator=(const HistogramBaggingRegressor&) = delete;

    HistogramBaggingRegressor(HistogramBaggingRegressor&&) = default;
    HistogramBaggingRegressor& operator=(HistogramBaggingRegressor&&) = default;

    void fit(const QuantizedDataset& dataset);
    double predict(const std::vector<double>& x) const;

private:
    std::vector<HistogramTreeRegressor> trees;
};