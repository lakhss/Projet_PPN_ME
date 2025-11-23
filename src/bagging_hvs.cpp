#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <random>
#include "decision_tree.hpp"

/**
 * @brief Load CSV file
 */
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

/**
 * @brief Bagging predictor
 */
double bagging_predict(const std::vector<Node*>& trees,
                       const std::vector<double>& sample)
{
    double sum = 0.0;
    for (auto tree : trees) {
        double log_pred = predict(tree, sample);
        sum += std::exp(log_pred) - 1e-6; // back to original scale
    }
    return sum / trees.size();
}

int main() {
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    load_csv("../datasets/15k_hvs.csv", X, y);

    std::cout << "Dataset loaded: " << X.size() 
              << " samples, " << X[0].size() << " features.\n";

    // Log-transform target
    std::vector<double> log_y = y;
    for (double &v : log_y)
        v = std::log(v + 1e-6);

    // --- Bagging setup ---
    const int N_TREES = 10; // number of trees
    const int MAX_DEPTH = 25;
    const int MIN_SAMPLES = 3;
    std::vector<Node*> trees;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, X.size() - 1);

    for (int t = 0; t < N_TREES; ++t) {
        // Sample dataset with replacement
        std::vector<std::vector<double>> X_sample;
        std::vector<double> y_sample;
        for (size_t i = 0; i < X.size(); ++i) {
            int idx = dis(gen);
            X_sample.push_back(X[idx]);
            y_sample.push_back(log_y[idx]);
        }

        // Build tree
        Node* tree = build_tree(X_sample, y_sample, 0, MAX_DEPTH, MIN_SAMPLES);
        trees.push_back(tree);
    }

    // --- Make predictions ---
    std::cout << "\n--- Bagging predictions ---\n";
    for (size_t i = 0; i < X.size(); i++) {
        double pred = bagging_predict(trees, X[i]);
        std::cout << "Line " << i+1
                  << " -> prediction = " << pred
                  << " | real = " << y[i] << "\n";
    }

    return 0;
}
