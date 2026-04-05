#pragma once

#include "Node.hpp"
#include "Matrix.hpp"
#include "Histogram.hpp"

#include <vector>
#include <string>
#include <fstream>
#include <tuple>
#include <memory>

class HistogramTreeRegressor {
public:
    std::unique_ptr<Node> root = nullptr;

    int max_depth = 10;
    std::size_t min_samples_split = 10;
    double min_gain = 1e-7;
    int n_bins = 16;   // nombre de bins pour l'histogramme

    HistogramTreeRegressor() = default;

    HistogramTreeRegressor(const HistogramTreeRegressor&) = delete;
    HistogramTreeRegressor& operator=(const HistogramTreeRegressor&) = delete;

    HistogramTreeRegressor(HistogramTreeRegressor&&) = default;
    HistogramTreeRegressor& operator=(HistogramTreeRegressor&&) = default;

    void fit(const Matrix& X, const std::vector<double>& y);
    double predict(const std::vector<double>& x) const;
    void print_tree(const Node* node = nullptr, int depth = 0);

    void export_to_dot(const std::string& filename) const;
    void export_node(std::ofstream& out, const Node* node, int& counter) const;

    ~HistogramTreeRegressor() = default;

private:
    std::unique_ptr<Node> build(const std::vector<std::size_t>& indices,
                                const Matrix& X,
                                const std::vector<double>& y,
                                int depth);

    HistogramSplit find_best_split_histogram(const std::vector<std::size_t>& indices,
                                             const Matrix& X,
                                             const std::vector<double>& y);

    double mse_from_stats(double sum, double sq_sum, std::size_t count) const;
};