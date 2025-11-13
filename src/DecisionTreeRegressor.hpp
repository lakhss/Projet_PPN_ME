#pragma once
#include "Node.hpp"
#include <vector>

class DecisionTreeRegressor {
public:
    Node* root = nullptr;
    int max_depth = 10;
    int min_samples_split = 10;
    double min_gain = 1e-7;

    void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y);
    double predict(const std::vector<double>& x) const;
    ~DecisionTreeRegressor() { delete root; }

private:
    Node* build(const std::vector<size_t>& indices,
                const std::vector<std::vector<double>>& X,
                const std::vector<double>& y,
                int depth);
    std::pair<int, double> find_best_split(const std::vector<size_t>& indices,
                                           const std::vector<std::vector<double>>& X,
                                           const std::vector<double>& y);
    double mse(const std::vector<double>& y);
};