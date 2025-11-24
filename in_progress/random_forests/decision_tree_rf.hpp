#ifndef DECISION_TREE_RF_HPP
#define DECISION_TREE_RF_HPP

#include <vector>
#include <random>

struct NodeRF {
    bool is_leaf = false;
    int samples = 0;
    int feature_index = -1;
    double threshold = 0.0;
    double value = 0.0;

    NodeRF* left = nullptr;
    NodeRF* right = nullptr;
};

double mean_rf(const std::vector<double>& values);
double mse_rf(const std::vector<double>& values);

NodeRF* build_tree_rf(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& y,
    int depth,
    int MAX_DEPTH,
    int MIN_SAMPLES,
    int MAX_FEATURES,
    std::mt19937& gen
);

double predict_rf(NodeRF* node, const std::vector<double>& sample);

#endif
