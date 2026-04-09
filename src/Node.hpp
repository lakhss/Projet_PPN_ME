#pragma once
#include <memory>

struct Node {
    double value = 0.0;
    int feature_idx = -1;
    double threshold = 0.0; 
    std::unique_ptr<Node> left = nullptr;
    std::unique_ptr<Node> right = nullptr;
    bool is_leaf = true;
};