#ifndef DECISION_TREE_REGRESSOR_HPP
#define DECISION_TREE_REGRESSOR_HPP

#include "Node.hpp"
#include <vector>
#include <string>
#include <fstream>

class DecisionTreeRegressor {
public:
    Node* root = nullptr;
    int max_depth = 10;
    int min_samples_split = 10;
    double min_gain = 1e-7;

    void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y);
    double predict(const std::vector<double>& x) const;
    void print_tree(Node* node, int depth = 0);

    // Ajout Graphviz
    void export_to_dot(const std::string& filename) const;
    void export_node(std::ofstream& out, Node* node, int& counter) const;

    ~DecisionTreeRegressor() { delete root; }

private:
    Node* build(const std::vector<size_t>& indices,
                const std::vector<std::vector<double>>& X,
                const std::vector<double>& y,
                int depth);

    std::tuple<int, double, double> find_best_split(const std::vector<size_t>& indices,
                                                    const std::vector<std::vector<double>>& X,
                                                    const std::vector<double>& y);

    double mse(const std::vector<double>& y);
};

#endif
