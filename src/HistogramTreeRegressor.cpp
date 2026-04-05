#include "HistogramTreeRegressor.hpp"

#include <algorithm>
#include <numeric>
#include <iostream>
#include <limits>
#include <cmath>

double HistogramTreeRegressor::mse_from_stats(double sum,
                                              double sq_sum,
                                              std::size_t count) const {
    if (count == 0) return 0.0;
    double mean = sum / static_cast<double>(count);
    return (sq_sum / static_cast<double>(count)) - mean * mean;
}

HistogramSplit HistogramTreeRegressor::find_best_split_histogram(
    const std::vector<std::size_t>& indices,
    const Matrix& X,
    const std::vector<double>& y) {

    HistogramSplit best_split;

    if (indices.empty() || X.cols() == 0) {
        return best_split;
    }

    double global_sum = 0.0;
    double global_sq_sum = 0.0;
    for (std::size_t i : indices) {
        global_sum += y[i];
        global_sq_sum += y[i] * y[i];
    }

    const double parent_mse = mse_from_stats(global_sum, global_sq_sum, indices.size());

    for (std::size_t feat = 0; feat < X.cols(); ++feat) {
        double min_val = std::numeric_limits<double>::max();
        double max_val = std::numeric_limits<double>::lowest();

        for (std::size_t i : indices) {
            double v = X(i, feat);
            if (v < min_val) min_val = v;
            if (v > max_val) max_val = v;
        }

        if (min_val == max_val) {
            continue;
        }

        std::vector<BinStats> hist(static_cast<std::size_t>(n_bins));
        const double width = (max_val - min_val) / static_cast<double>(n_bins);

        for (std::size_t i : indices) {
            double v = X(i, feat);
            int bin = static_cast<int>((v - min_val) / width);

            if (bin < 0) bin = 0;
            if (bin >= n_bins) bin = n_bins - 1;

            hist[static_cast<std::size_t>(bin)].count++;
            hist[static_cast<std::size_t>(bin)].sum += y[i];
            hist[static_cast<std::size_t>(bin)].sq_sum += y[i] * y[i];
        }

        double left_sum = 0.0;
        double left_sq_sum = 0.0;
        std::size_t left_count = 0;

        double right_sum = global_sum;
        double right_sq_sum = global_sq_sum;
        std::size_t right_count = indices.size();

        for (int b = 0; b < n_bins - 1; ++b) {
            const BinStats& current = hist[static_cast<std::size_t>(b)];

            left_count += current.count;
            left_sum += current.sum;
            left_sq_sum += current.sq_sum;

            right_count -= current.count;
            right_sum -= current.sum;
            right_sq_sum -= current.sq_sum;

            if (left_count < min_samples_split || right_count < min_samples_split) {
                continue;
            }

            double mse_left = mse_from_stats(left_sum, left_sq_sum, left_count);
            double mse_right = mse_from_stats(right_sum, right_sq_sum, right_count);

            double weighted_mse =
                (static_cast<double>(left_count) * mse_left +
                 static_cast<double>(right_count) * mse_right)
                / static_cast<double>(indices.size());

            double gain = parent_mse - weighted_mse;

            if (gain > best_split.gain) {
                best_split.gain = gain;
                best_split.feature_idx = static_cast<int>(feat);
                best_split.threshold = min_val + (static_cast<double>(b + 1) * width);
            }
        }
    }

    return best_split;
}

std::unique_ptr<Node> HistogramTreeRegressor::build(
    const std::vector<std::size_t>& indices,
    const Matrix& X,
    const std::vector<double>& y,
    int depth) {

    auto node = std::make_unique<Node>();

    double mean = 0.0;
    for (std::size_t i : indices) {
        mean += y[i];
    }
    if (!indices.empty()) {
        mean /= static_cast<double>(indices.size());
    }

    node->value = mean;

    if (depth >= max_depth || indices.size() < min_samples_split) {
        node->is_leaf = true;
        return node;
    }

    HistogramSplit split = find_best_split_histogram(indices, X, y);

    if (split.feature_idx == -1 || split.gain < min_gain) {
        node->is_leaf = true;
        return node;
    }

    std::vector<std::size_t> left_idx, right_idx;
    left_idx.reserve(indices.size());
    right_idx.reserve(indices.size());

    for (std::size_t i : indices) {
        if (X(i, static_cast<std::size_t>(split.feature_idx)) <= split.threshold) {
            left_idx.push_back(i);
        } else {
            right_idx.push_back(i);
        }
    }

    if (left_idx.empty() || right_idx.empty()) {
        node->is_leaf = true;
        return node;
    }

    node->is_leaf = false;
    node->feature_idx = split.feature_idx;
    node->threshold = split.threshold;

    node->left = build(left_idx, X, y, depth + 1);
    node->right = build(right_idx, X, y, depth + 1);

    return node;
}

void HistogramTreeRegressor::fit(const Matrix& X,
                                 const std::vector<double>& y) {
    std::vector<std::size_t> indices(X.rows());
    std::iota(indices.begin(), indices.end(), 0);
    root = build(indices, X, y, 0);
}

double HistogramTreeRegressor::predict(const std::vector<double>& x) const {
    Node* node = root.get();

    while (node && !node->is_leaf) {
        if (x[static_cast<std::size_t>(node->feature_idx)] <= node->threshold) {
            node = node->left.get();
        } else {
            node = node->right.get();
        }
    }

    return node ? node->value : 0.0;
}

void HistogramTreeRegressor::print_tree(const Node* node, int depth) {
    if (!node) {
        if (depth == 0 && root) node = root.get();
        else return;
    }

    std::string indent(static_cast<std::size_t>(depth * 2), ' ');

    if (node->is_leaf) {
        std::cout << indent << "Feuille: value = " << node->value << std::endl;
    } else {
        std::cout << indent << "Split: feature " << node->feature_idx
                  << " <= " << node->threshold << std::endl;
    }

    if (node->left) print_tree(node->left.get(), depth + 1);
    if (node->right) print_tree(node->right.get(), depth + 1);
}

void HistogramTreeRegressor::export_to_dot(const std::string& filename) const {
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

    std::cout << "Arbre histogramme exporté dans " << filename << std::endl;
}

void HistogramTreeRegressor::export_node(std::ofstream& out,
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
            << node->threshold << "\", fillcolor=\"#ffcc99\"];\n";
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