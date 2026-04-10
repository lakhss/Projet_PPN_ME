#pragma once

#include "QuantizedDataset.hpp" 
#include "Histogram.hpp"

#include <vector>
#include <string>
#include <fstream>
#include <memory>
#include <cstdint>

// new node pour SoA comme on change pas l'ancien sinon le naive va pas fonctionner 
struct alignas(32) HPCNode {
    double value = 0.0;          // 8 octets
    int feature_idx = -1;        // 4 octets
    int left_child = -1;         // 4 octets
    int right_child = -1;        // 4 octets
    uint8_t threshold_bin = 0;   // 1 octet
    bool is_leaf = true;         // 1 octet
    float split_value = 0.0f;    // 4 octets 
};

class HistogramTreeRegressor {
public: 
    
    int max_depth = 10;
    std::size_t min_samples_split = 10;
    double min_gain = 1e-7;
    int n_bins = 256; // uint8_t pour l'utiliser a fond 

    HistogramTreeRegressor() = default;
    ~HistogramTreeRegressor() = default;

    HistogramTreeRegressor(const HistogramTreeRegressor&) = delete;
    HistogramTreeRegressor& operator=(const HistogramTreeRegressor&) = delete;

    HistogramTreeRegressor(HistogramTreeRegressor&&) = default;
    HistogramTreeRegressor& operator=(HistogramTreeRegressor&&) = default;

    void fit(const QuantizedDataset& dataset);
    double predict(const std::vector<double>& x) const;
    
    void fit_bootstrap(const QuantizedDataset& dataset, std::vector<std::size_t>& bootstrap_indices); 

    void print_tree(int node_idx = -1, int depth = 0) const;
    void export_to_dot(const std::string& filename) const;

private: 
    
    std::vector<HPCNode> tree_nodes_; 
    int root_idx_ = -1;               

    int build(std::vector<std::size_t>& indices,
              std::size_t start,
              std::size_t end,
              int depth,
              const QuantizedDataset& dataset);

    HistogramSplit find_best_split_histogram(const std::vector<std::size_t>& indices,
                                             std::size_t start,
                                             std::size_t end,
                                             const QuantizedDataset& dataset);

    double mse_from_stats(double sum, double sq_sum, std::size_t count) const;
    
    void export_node(std::ofstream& out, int node_idx, int& counter) const;
};