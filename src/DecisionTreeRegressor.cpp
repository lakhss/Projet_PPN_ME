#include "DecisionTreeRegressor.hpp"
#include <algorithm>
#include <numeric>
#include <random>
#include <iostream>

double DecisionTreeRegressor::mse(const std::vector<double>& y) {
    if (y.size() <= 1) return 0.0;
    double mean = std::accumulate(y.begin(), y.end(), 0.0) / y.size();
    double sum = 0.0;
    for (double v : y) sum += (v - mean) * (v - mean);
    return sum / y.size();
}

std::tuple<int, double, double> DecisionTreeRegressor::find_best_split(
    const std::vector<size_t>& indices,
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& y) {

    int best_feat = -1;
    double best_thr = 0.0;
    double best_gain = 0.0;
    double parent_mse = mse(y);
    int n = indices.size();
    int n_features = X[0].size();

    for (int feat = 0; feat < n_features; ++feat) {
        std::vector<std::pair<double, size_t>> pairs;
        for (size_t i : indices)
            pairs.emplace_back(X[i][feat], i);
        std::sort(pairs.begin(), pairs.end());

        for (size_t i = 1; i < pairs.size(); ++i) {
            if (pairs[i].first == pairs[i-1].first) continue;
            double thr = (pairs[i-1].first + pairs[i].first) / 2.0;

            std::vector<double> left_y, right_y;
            for (const auto& p : pairs) {
                if (p.first <= thr) left_y.push_back(y[p.second]);
                else right_y.push_back(y[p.second]);
            }

            if (left_y.size() < min_samples_split || right_y.size() < min_samples_split)
                continue;

            double weighted_mse = (left_y.size() * mse(left_y) + right_y.size() * mse(right_y)) / n;
            double gain = parent_mse - weighted_mse;

            if (gain > best_gain) {
                best_gain = gain;
                best_feat = feat;
                best_thr = thr;
            }
        }
    }
    return {best_feat, best_thr, best_gain};  // Ajout de best_gain ici
}

Node* DecisionTreeRegressor::build(const std::vector<size_t>& indices,
                                   const std::vector<std::vector<double>>& X,
                                   const std::vector<double>& y,
                                   int depth) {
    Node* node = new Node();

    // double mean = std::accumulate(y.begin(), y.end(), 0.0) / y.size();
    double mean = 0 
    for (int i :indices) {
        mean += y[i];
    }
    mean /= indices.size();

    node->value = mean;

    if (depth >= max_depth || indices.size() < min_samples_split) {
        node->is_leaf = true;
        return node;
    }

    auto [feat, thr, gain] = find_best_split(indices, X, y);  // Décomposition avec gain
    if (feat == -1 || gain < min_gain) {
        node->is_leaf = true;
        return node;
    }

    node->is_leaf = false;
    node->feature_idx = feat;
    node->threshold = thr;

    std::vector<size_t> left_idx, right_idx;
    for (size_t i : indices) {
        if (X[i][feat] <= thr) left_idx.push_back(i);
        else right_idx.push_back(i);
    }

    node->left = build(left_idx, X, y, depth + 1);
    node->right = build(right_idx, X, y, depth + 1);
    return node;
}

void DecisionTreeRegressor::fit(const std::vector<std::vector<double>>& X,
                                const std::vector<double>& y) {
    std::vector<size_t> indices(X.size());
    std::iota(indices.begin(), indices.end(), 0);
    root = build(indices, X, y, 0);
    std::cout << "Arbre entraîné (profondeur max = " << max_depth << ")" << std::endl;
}

double DecisionTreeRegressor::predict(const std::vector<double>& x) const {
    Node* node = root;
    while (!node->is_leaf) {
        if (x[node->feature_idx] <= node->threshold)
            node = node->left;
        else
            node = node->right;
    }
    return node->value;
}

void DecisionTreeRegressor::print_tree(Node* node, int depth) {
    if (!node) return;
    std::string indent(depth * 2, ' ');  // Espace pour l'indentation
    if (node->is_leaf) {
        std::cout << indent << "Feuille: value = " << node->value << std::endl;
    } else {
        std::cout << indent << "Split: feature " << node->feature_idx 
                  << " <= " << node->threshold << std::endl;
    }
    print_tree(node->left, depth + 1);
    print_tree(node->right, depth + 1);
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
    export_node(out, root, counter);

    out << "}\n";
    out.close();

    std::cout << "Arbre exporté dans " << filename << std::endl;
}

void DecisionTreeRegressor::export_node(std::ofstream& out, Node* node, int& counter) const {
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
        export_node(out, node->left, counter);
        out << "node" << id << " -> node" << left_id << " [label=\"yes\"];\n";

        int right_id = counter;
        export_node(out, node->right, counter);
        out << "node" << id << " -> node" << right_id << " [label=\"no\"];\n";
    }
}
