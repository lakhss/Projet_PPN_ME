#include "DecisionTreeRegressor.hpp"
#include "Matrix.hpp"

#include <algorithm>
#include <numeric>
#include <random>
#include <iostream>

/**
 * @brief Calculate Mean Squared Error of given targets
 * @param y Target values
 * @return MSE value
 */
double DecisionTreeRegressor::mse(const std::vector<double>& y) {
    if (y.size() <= 1) return 0.0;

    double mean = std::accumulate(y.begin(), y.end(), 0.0) / y.size();
    double sum = 0.0;
    for (double v : y) {
        sum += (v - mean) * (v - mean);
    }
    return sum / y.size();
}

/**
 * @brief Find the best feature and threshold to split the node
 * @param indices Indices of samples in the current node
 * @param X Feature matrix
 * @param y Target values
 * @return tuple (best_feature_index, best_threshold, best_gain)
 */
std::tuple<int, double, double> DecisionTreeRegressor::find_best_split(
    const std::vector<size_t>& indices,
    const Matrix& X,
    const std::vector<double>& y) {

    int best_feat = -1;
    double best_thr = 0.0;
    double best_gain = -1.0;

    double global_sum = 0.0;
    double global_sq_sum = 0.0;
    for (size_t i : indices) {
        global_sum += y[i];
        global_sq_sum += y[i] * y[i];
    }

    double parent_mse =
        (global_sq_sum / indices.size()) -
        (global_sum / indices.size()) * (global_sum / indices.size());

    int n_features = static_cast<int>(X.cols());

    for (int feat = 0; feat < n_features; ++feat) {
        std::vector<std::pair<double, size_t>> pairs;
        pairs.reserve(indices.size());

        for (size_t i : indices) {
            pairs.emplace_back(X(i, feat), i);
        }

        std::sort(pairs.begin(), pairs.end());

        double left_sum = 0.0;
        double left_sq_sum = 0.0;
        double right_sum = global_sum;
        double right_sq_sum = global_sq_sum;
        size_t left_count = 0;
        size_t right_count = indices.size();

        for (size_t i = 0; i + 1 < pairs.size(); ++i) {
            double val_y = y[pairs[i].second];

            left_sum += val_y;
            left_sq_sum += val_y * val_y;
            left_count++;

            right_sum -= val_y;
            right_sq_sum -= val_y * val_y;
            right_count--;

            if (pairs[i].first == pairs[i + 1].first) continue;
            if (left_count < static_cast<size_t>(min_samples_split) ||
                right_count < static_cast<size_t>(min_samples_split)) continue;

            double mse_left =
                (left_count == 0) ? 0.0 :
                (left_sq_sum / left_count) -
                (left_sum / left_count) * (left_sum / left_count);

            double mse_right =
                (right_count == 0) ? 0.0 :
                (right_sq_sum / right_count) -
                (right_sum / right_count) * (right_sum / right_count);

            double weighted_mse =
                (left_count * mse_left + right_count * mse_right) / indices.size();

            double gain = parent_mse - weighted_mse;

            if (gain > best_gain) {
                best_gain = gain;
                best_feat = feat;
                best_thr = (pairs[i].first + pairs[i + 1].first) / 2.0;
            }
        }
    }

    return {best_feat, best_thr, best_gain};
}

std::unique_ptr<Node> DecisionTreeRegressor::build(
    const std::vector<size_t>& indices,
    const Matrix& X,
    const std::vector<double>& y,
    int depth) {

    auto node = std::make_unique<Node>();

    double mean = 0.0;
    for (size_t i : indices) {
        mean += y[i];
    }
    if (!indices.empty()) {
        mean /= indices.size();
    }

    node->value = mean;

    if (depth >= max_depth || indices.size() < static_cast<size_t>(min_samples_split)) {
        node->is_leaf = true;
        return node;
    }

    auto [feat, thr, gain] = find_best_split(indices, X, y);

    if (feat == -1 || gain < min_gain) {
        node->is_leaf = true;
        return node;
    }

    node->is_leaf = false;
    node->feature_idx = feat;
    node->threshold = thr;

    std::vector<size_t> left_idx, right_idx;
    left_idx.reserve(indices.size());
    right_idx.reserve(indices.size());

    for (size_t i : indices) {
        if (X(i, feat) <= thr) left_idx.push_back(i);
        else right_idx.push_back(i);
    }

    node->left = build(left_idx, X, y, depth + 1);
    node->right = build(right_idx, X, y, depth + 1);

    return node;
}

void DecisionTreeRegressor::fit(const Matrix& X,
                                const std::vector<double>& y) {
    std::vector<size_t> indices(X.rows());
    std::iota(indices.begin(), indices.end(), 0);
    root = build(indices, X, y, 0);
}

double DecisionTreeRegressor::predict(const std::vector<double>& x) const {
    Node* node = root.get();

    while (node && !node->is_leaf) {
        if (x[node->feature_idx] <= node->threshold)
            node = node->left.get();
        else
            node = node->right.get();
    }

    return node ? node->value : 0.0;
}

void DecisionTreeRegressor::print_tree(const Node* node, int depth) {
    if (!node) {
        if (depth == 0 && root) node = root.get();
        else return;
    }

    std::string indent(depth * 2, ' ');
    if (node->is_leaf) {
        std::cout << indent << "Feuille: value = " << node->value << std::endl;
    } else {
        std::cout << indent << "Split: feature " << node->feature_idx
                  << " <= " << node->threshold << std::endl;
    }

    if (node->left) print_tree(node->left.get(), depth + 1);
    if (node->right) print_tree(node->right.get(), depth + 1);
}

void DecisionTreeRegressor::export_to_dot(const std::string& filename) const {
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "Impossible d'écrire dans " << filename << std::endl;
        return;
    }

    out << "digraph Tree {\n";
    out << "node [shape=box, style=filled, fontsize=10];\n";

    int counter = 0;
    export_node(out, root.get(), counter);

    out << "}\n";
    out.close();

    std::cout << "Arbre exporté dans " << filename << std::endl;
}

void DecisionTreeRegressor::export_node(std::ofstream& out,
                                        const Node* node,
                                        int& counter) const {
    if (!node) return;

    int id = counter++;

    if (node->is_leaf) {
        out << "node" << id << " [label=\"Leaf\\nvalue = "
            << node->value << "\", fillcolor=\"#aaffaa\"];\n";
    } else {
        out << "node" << id
            << " [label=\"X[" << node->feature_idx << "] <= "
            << node->threshold << "\", fillcolor=\"#aaaaff\"];\n";
    }

    if (!node->is_leaf) {
        int left_id = counter;
        if (node->left) export_node(out, node->left.get(), counter);
        out << "node" << id << " -> node" << left_id << " [label=\"yes\"];\n";

        int right_id = counter;
        if (node->right) export_node(out, node->right.get(), counter);
        out << "node" << id << " -> node" << right_id << " [label=\"no\"];\n";
    }
}