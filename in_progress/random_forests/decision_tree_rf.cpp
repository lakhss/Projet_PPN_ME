#include "decision_tree_rf.hpp"
#include <algorithm>
#include <cmath>

#define MSE_MAX_RF 1e12

double mean_rf(const std::vector<double>& values) {
    double s = 0;
    for (double v : values) s += v;
    return s / values.size();
}

double mse_rf(const std::vector<double>& values) {
    if (values.empty()) return 0.0;
    double m = mean_rf(values);
    double s = 0;
    for (double v : values) {
        double d = v - m;
        s += d * d;
    }
    return s / values.size();
}

NodeRF* build_tree_rf(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& y,
    int depth,
    int MAX_DEPTH,
    int MIN_SAMPLES,
    int MAX_FEATURES,
    std::mt19937& gen
) {
    NodeRF* node = new NodeRF();
    node->samples = y.size();

    double current_mse = mse_rf(y);

    if (depth >= MAX_DEPTH || y.size() <= MIN_SAMPLES || current_mse < 1e-6) {
        node->is_leaf = true;
        node->value = mean_rf(y);
        return node;
    }

    int n = X.size();
    int m = X[0].size();

    // === Select random subset of features ===
    std::vector<int> all_features(m);
    for (int i = 0; i < m; i++) all_features[i] = i;

    std::shuffle(all_features.begin(), all_features.end(), gen);
    std::vector<int> chosen_features(
        all_features.begin(),
        all_features.begin() + std::min(MAX_FEATURES, m)
    );

    double best_mse = MSE_MAX_RF;
    int best_feature = -1;
    double best_threshold = 0.0;

    for (int feature : chosen_features) {
        std::vector<double> sorted_values;
        sorted_values.reserve(n);

        for (auto &row : X) sorted_values.push_back(row[feature]);
        std::sort(sorted_values.begin(), sorted_values.end());

        for (int i = 0; i < n - 1; ++i) {
            double thr = (sorted_values[i] + sorted_values[i+1]) / 2;

            std::vector<double> left_y, right_y;

            for (int k = 0; k < n; k++) {
                if (X[k][feature] <= thr) left_y.push_back(y[k]);
                else right_y.push_back(y[k]);
            }

            if (left_y.empty() || right_y.empty()) continue;

            double mse_split =
                (left_y.size() * mse_rf(left_y) +
                 right_y.size() * mse_rf(right_y)) / n;

            if (mse_split < best_mse) {
                best_mse = mse_split;
                best_feature = feature;
                best_threshold = thr;
            }
        }
    }

    if (best_feature == -1) {
        node->is_leaf = true;
        node->value = mean_rf(y);
        return node;
    }

    node->feature_index = best_feature;
    node->threshold = best_threshold;

    std::vector<std::vector<double>> X_left, X_right;
    std::vector<double> y_left, y_right;

    for (int i = 0; i < n; i++) {
        if (X[i][best_feature] <= best_threshold) {
            X_left.push_back(X[i]);
            y_left.push_back(y[i]);
        } else {
            X_right.push_back(X[i]);
            y_right.push_back(y[i]);
        }
    }

    node->left = build_tree_rf(
        X_left, y_left,
        depth + 1, MAX_DEPTH, MIN_SAMPLES,
        MAX_FEATURES,
        gen
    );

    node->right = build_tree_rf(
        X_right, y_right,
        depth + 1, MAX_DEPTH, MIN_SAMPLES,
        MAX_FEATURES,
        gen
    );

    return node;
}

double predict_rf(NodeRF* node, const std::vector<double>& sample) {
    if (node->is_leaf) return node->value;

    if (sample[node->feature_index] <= node->threshold)
        return predict_rf(node->left, sample);

    return predict_rf(node->right, sample);
}
