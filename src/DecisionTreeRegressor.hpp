#ifndef DECISION_TREE_REGRESSOR_HPP
#define DECISION_TREE_REGRESSOR_HPP

#include "Node.hpp"
#include "Matrix.hpp"

#include <vector>
#include <string>
#include <fstream>
#include <tuple>
#include <memory>

class DecisionTreeRegressor {
public:
    std::unique_ptr<Node> root = nullptr;

    int max_depth = 10;
    size_t min_samples_split = 10;
    double min_gain = 1e-7;

    DecisionTreeRegressor() = default;

    // copie interdite
    DecisionTreeRegressor(const DecisionTreeRegressor&) = delete;
    DecisionTreeRegressor& operator=(const DecisionTreeRegressor&) = delete;

    // déplacement autorisé
    DecisionTreeRegressor(DecisionTreeRegressor&&) = default;
    DecisionTreeRegressor& operator=(DecisionTreeRegressor&&) = default;

    void fit(const Matrix& X, const std::vector<double>& y);
    double predict(const std::vector<double>& x) const;
    void print_tree(const Node* node, int depth = 0);

    void export_to_dot(const std::string& filename) const;
    void export_node(std::ofstream& out, const Node* node, int& counter) const;

    ~DecisionTreeRegressor() = default;

private:
    std::unique_ptr<Node> build(const std::vector<size_t>& indices,
                                const Matrix& X,
                                const std::vector<double>& y,
                                int depth);

    std::tuple<int, double, double> find_best_split(const std::vector<size_t>& indices,
                                                    const Matrix& X,
                                                    const std::vector<double>& y);

    double mse(const std::vector<double>& y);
};

#endif