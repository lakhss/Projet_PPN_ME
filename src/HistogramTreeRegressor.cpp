#include "HistogramTreeRegressor.hpp"
#include "QuantizedDataset.hpp"

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
    std::size_t start,
    std::size_t end,
    const QuantizedDataset& dataset) { // nouvelle structure SoA avec Quantized

    HistogramSplit best_split;
    std::size_t node_samples = end - start;

    if (node_samples < min_samples_split || dataset.get_num_features() == 0) {
        return best_split;
    }

    double global_sum = 0.0;
    double global_sq_sum = 0.0;
    const auto& y = dataset.get_targets();

    for (std::size_t i = start; i < end; ++i) {
        double target = y[indices[i]];
        global_sum += target;
        global_sq_sum += target * target;
    }

    const double parent_mse = mse_from_stats(global_sum, global_sq_sum, node_samples);
    const size_t num_features = dataset.get_num_features();

    for (std::size_t feat = 0; feat < num_features; ++feat) {
        
        FastBinStats hist[256] = {}; 
        
        // onrecup vers le tableau contigu (SoA)
        const auto& feature_col = dataset.get_feature_column(feat);

        for (std::size_t i = start; i < end; ++i) {
            std::size_t idx = indices[i];
            uint8_t bin = feature_col[idx];  // en theorie ca va lire en flux lineaire 
            double target = y[idx];

            hist[bin].count++;
            hist[bin].sum += target;
            hist[bin].sq_sum += target * target;
        }

        double left_sum = 0.0;
        double left_sq_sum = 0.0;
        std::size_t left_count = 0;

        double right_sum = global_sum;
        double right_sq_sum = global_sq_sum;
        std::size_t right_count = node_samples;

        // On itere sur les 255 coupures possibles (de gauche à droite) qui est le max bin 
        for (int b = 0; b < 255; ++b) {
            const FastBinStats& current = hist[b];

            // S'il n'y a aucun individu dans ce bin, on passe au suivant
            if (current.count == 0) continue; 

            left_count += current.count;
            left_sum += current.sum;
            left_sq_sum += current.sq_sum;

            right_count -= current.count;
            right_sum -= current.sum;
            right_sq_sum -= current.sq_sum;

            // critère d'arret rapide pour éviter les divisions useless
            if (left_count < min_samples_split || right_count < min_samples_split) {
                continue;
            }

            double mse_left = mse_from_stats(left_sum, left_sq_sum, left_count);
            double mse_right = mse_from_stats(right_sum, right_sq_sum, right_count);

            double weighted_mse = 
                (static_cast<double>(left_count) * mse_left + 
                 static_cast<double>(right_count) * mse_right) 
                / static_cast<double>(node_samples);

            double gain = parent_mse - weighted_mse;

            // Mise a jour si le gain est strictement supeerieur
            if (gain > best_split.gain) {
                best_split.gain = gain;
                best_split.feature_idx = static_cast<int>(feat);
                best_split.threshold_bin = static_cast<uint8_t>(b);
            }
        }
    }

    return best_split;
}

#include <algorithm> 

int HistogramTreeRegressor::build(
    std::vector<std::size_t>& indices,
    std::size_t start,
    std::size_t end,
    int depth,
    const QuantizedDataset& dataset) {

    // Allocation depuis le Pool
    int node_idx = static_cast<int>(tree_nodes_.size());
    tree_nodes_.emplace_back(); 
    
    // au lieu d'utiliser matrix et Node la on fait son propre node a l'interieur de build  
    HPCNode& current_node = tree_nodes_[node_idx]; 

    std::size_t node_samples = end - start;

    double mean = 0.0;
    const auto& y = dataset.get_targets();
    for (std::size_t i = start; i < end; ++i) {
        mean += y[indices[i]];
    }
    if (node_samples > 0) {
        mean /= static_cast<double>(node_samples);
    }
    current_node.value = mean;

    if (depth >= max_depth || node_samples < min_samples_split) {
        current_node.is_leaf = true;
        return node_idx;
    }

    HistogramSplit split = find_best_split_histogram(indices, start, end, dataset);

    if (split.feature_idx == -1 || split.gain < min_gain) {
        current_node.is_leaf = true;
        return node_idx;
    }

    current_node.is_leaf = false;
    current_node.feature_idx = split.feature_idx;
    current_node.threshold_bin = split.threshold_bin;

    const auto& meta = dataset.get_meta(split.feature_idx);
    current_node.split_value = meta.min_val + static_cast<float>(split.threshold_bin + 1) * meta.bin_width;

    const auto& feature_col = dataset.get_feature_column(split.feature_idx);
    uint8_t threshold = split.threshold_bin;

    auto bound_it = std::partition(
        indices.begin() + start, 
        indices.begin() + end,
        [&feature_col, threshold](std::size_t idx) {
            return feature_col[idx] <= threshold;
        }
    );

    std::size_t bound = std::distance(indices.begin(), bound_it);

    if (bound == start || bound == end) {
        current_node.is_leaf = true;
        return node_idx;
    }

    current_node.left_child = build(indices, start, bound, depth + 1, dataset);
    current_node.right_child = build(indices, bound, end, depth + 1, dataset);

    return node_idx;
}

void HistogramTreeRegressor::fit(const QuantizedDataset& dataset) {
    std::size_t num_samples = dataset.get_num_samples();
    
    // vecteur global des indices pour le partitionnement In-Place
    std::vector<std::size_t> indices(num_samples);
    std::iota(indices.begin(), indices.end(), 0);

    // vn arbre binaire parfait de profondeur D possède (2^(D+1) - 1) nœuds merci APP si je ne me trompe pas
    // Pour une profondeur de 10 ca fait  2047 nœuds
    std::size_t max_nodes = (1ULL << (max_depth + 1)) - 1;
    tree_nodes_.clear();
    tree_nodes_.reserve(max_nodes);

    root_idx_ = build(indices, 0, num_samples, 0, dataset);
}

void HistogramTreeRegressor::fit_bootstrap(const QuantizedDataset& dataset, std::vector<std::size_t>& bootstrap_indices) {
    tree_nodes_.clear();
    tree_nodes_.reserve( (1 << (max_depth + 1)) - 1 );

    root_idx_ = build(bootstrap_indices, 0, bootstrap_indices.size(), 0, dataset);
}

double HistogramTreeRegressor::predict(const std::vector<double>& x) const {
    if (tree_nodes_.empty() || root_idx_ == -1) return 0.0;

    int current_idx = root_idx_;
    
    while (current_idx != -1) {
        const HPCNode& node = tree_nodes_[current_idx];
        
        if (node.is_leaf) {
            return node.value;
        }

        if (x[node.feature_idx] <= node.split_value) {
            current_idx = node.left_child;
        } else {
            current_idx = node.right_child;
        }
    }

    return 0.0;
}

void HistogramTreeRegressor::print_tree(int node_idx, int depth) const {
    if (tree_nodes_.empty() || root_idx_ == -1) return;
    
    if (node_idx == -1) {
        node_idx = root_idx_;
    }

    const HPCNode& node = tree_nodes_[node_idx];
    std::string indent(static_cast<std::size_t>(depth * 2), ' ');

    if (node.is_leaf) {
        std::cout << indent << "Feuille: value = " << node.value << std::endl;
    } else {
        std::cout << indent << "Split: feature " << node.feature_idx
                  << " <= " << node.split_value 
                  << " (Bin: " << (int)node.threshold_bin << ")" << std::endl;
    }

    if (node.left_child != -1) print_tree(node.left_child, depth + 1);
    if (node.right_child != -1) print_tree(node.right_child, depth + 1);
}

void HistogramTreeRegressor::export_to_dot(const std::string& filename) const {
    if (tree_nodes_.empty() || root_idx_ == -1) return;

    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "Impossible d'écrire dans " << filename << std::endl;
        return;
    }

    out << "digraph Tree {\n";
    out << "node [shape=box, style=filled, fontsize=10];\n";

    int counter = 0;
    export_node(out, root_idx_, counter);

    out << "}\n";
    out.close();

    std::cout << "Arbre histogramme exporté dans " << filename << std::endl;
}

void HistogramTreeRegressor::export_node(std::ofstream& out, int node_idx, int& counter) const {
    if (node_idx == -1) return;

    const HPCNode& node = tree_nodes_[node_idx];
    int current_id = counter++;

    if (node.is_leaf) {
        out << "node" << current_id << " [label=\"Leaf\\nvalue = "
            << node.value << "\", fillcolor=\"#aaffaa\"];\n";
    } else {
        out << "node" << current_id
            << " [label=\"X[" << node.feature_idx << "] <= "
            << node.split_value << "\", fillcolor=\"#ffcc99\"];\n";
    }

    if (!node.is_leaf) {
        if (node.left_child != -1) {
            int left_id = counter;
            export_node(out, node.left_child, counter);
            out << "node" << current_id << " -> node" << left_id << " [label=\"yes\"];\n";
        }
        if (node.right_child != -1) {
            int right_id = counter;
            export_node(out, node.right_child, counter);
            out << "node" << current_id << " -> node" << right_id << " [label=\"no\"];\n";
        }
    }
}