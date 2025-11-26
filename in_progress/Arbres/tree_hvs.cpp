#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include "decision_tree.hpp"
#include "../../src/DataLoader.hpp"

int main() {
    std::vector<std::vector<double>> X;
    std::vector<double> y;

    DataLoader::load_csv("../../datasets/15k_hvs.csv", X, y);

    std::cout << "Dataset loaded: " << X.size() 
              << " samples, " << X[0].size() 
              << " features.\n";

    // --- Log-transform the target to stabilize variance ---
    std::vector<double> log_y = y;
    for (double &v : log_y)
        v = std::log(v + 1e-6); // Add epsilon to avoid log(0)

    
    Node* tree = build_tree(X, log_y, 0, 40, 1);

    std::cout << "\n--- HVS Predictions (log-transformed) ---\n";
    for (size_t i = 0; i < X.size(); i++) {
        double log_pred = predict(tree, X[i]);
        double pred = std::exp(log_pred) - 1e-6; // Back to original scale

        std::cout << "Line " << i+1
                  << " -> prediction = " << pred
                  << " | actual = " << y[i] << "\n";
    }

    return 0;
}
