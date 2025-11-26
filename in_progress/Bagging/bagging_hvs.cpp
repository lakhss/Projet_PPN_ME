#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <random>

#include "../random_forests/decision_tree_rf.hpp"

// ------------------------------
// Load CSV
// ------------------------------
void load_csv(const std::string& filename,
              std::vector<std::vector<double>>& X,
              std::vector<double>& y)
{
    std::ifstream file(filename);
    std::string line;
    std::getline(file, line); // skip header

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<double> row;
        double value;

        while (ss >> value) {
            row.push_back(value);
            if (ss.peek() == ',') ss.ignore();
        }

        if (row.empty()) continue;

        y.push_back(row.back());
        row.pop_back();
        X.push_back(row);
    }
}

// ------------------------------
// Bagging prediction (RF trees)
// ------------------------------
double bagging_predict(const std::vector<NodeRF*>& trees,
                       const std::vector<double>& sample)
{
    double sum = 0.0;
    for (auto t : trees) {
        double pred = predict_rf(t, sample);
        sum += std::exp(pred) - 1e-6;   // back-transform log
    }
    return sum / trees.size();
}


int main() {

    std::vector<std::vector<double>> X;
    std::vector<double> y;
    load_csv("../datasets/data_test.csv", X, y);

    std::cout << "Dataset loaded: " << X.size()
              << " samples, " << X[0].size() << " features.\n";

    // log-transform target
    std::vector<double> log_y = y;
    for (double &v : log_y) v = std::log(v + 1e-6);

    // Random Forest parameters
    const int N_TREES = 6;
    const int MAX_DEPTH = 12;
    const int MIN_SAMPLES = 3;
    const int MAX_FEATURES = 4;   // random subset of features per split

    std::vector<NodeRF*> trees;
    trees.reserve(N_TREES);

    // Random generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, X.size() - 1);

    // ------------------------------
    // Build each tree (Bootstrap + Random features)
    // ------------------------------
    for (int t = 0; t < N_TREES; t++) {

        std::vector<std::vector<double>> Xb;
        std::vector<double> yb;

        // bootstrap sampling
        for (size_t i = 0; i < X.size(); i++) {
            int idx = dis(gen);
            Xb.push_back(X[idx]);
            yb.push_back(log_y[idx]);
        }

        NodeRF* tree = build_tree_rf(
            Xb, yb,
            0,
            MAX_DEPTH,
            MIN_SAMPLES,
            MAX_FEATURES,
            gen
        );

        trees.push_back(tree);
    }

    // ------------------------------
    // Predictions
    // ------------------------------
    std::cout << "\n--- Random Forest predictions ---\n";

    for (size_t i = 0; i < X.size(); i++) {
        double pred = bagging_predict(trees, X[i]);
        std::cout << "Line " << i+1
                  << " -> prediction = " << pred
                  << " | real = " << y[i] << "\n";
    }

    return 0;
}
