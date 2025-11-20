#ifndef DECISION_TREE_HPP
#define DECISION_TREE_HPP

#include <vector>

struct Node {
    bool is_leaf = false;
    int samples = 0;
    int feature_index = -1;
    double threshold = 0.0;
    double value = 0.0;

    Node* left = nullptr;
    Node* right = nullptr;
};

double mean(const std::vector<double>& values);
double mse(const std::vector<double>& values);
Node* build_tree(const std::vector<std::vector<double>>& X,
                 const std::vector<double>& y,
                 int depth = 0,
                 int MAX_DEPTH = 10,
                 int MIN_SAMPLES = 3);
double predict(Node* node, const std::vector<double>& sample);

#endif
