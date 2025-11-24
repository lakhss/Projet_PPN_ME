#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include "decision_tree.hpp"
#include "DataLoader.hpp"

/**
 * @brief Load CSV file into features X and target y
 * @param filename Path to CSV file
 * @param X Output: feature matrix
 * @param y Output: target vector
 */
void load_csv(const std::string& filename,
              std::vector<std::vector<double>>& X,
              std::vector<double>& y)
{
    std::ifstream file(filename);
    std::string line;

    // Skip header
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<double> row;
        double value;

        while (ss >> value) {
            row.push_back(value);
            if (ss.peek() == ',') ss.ignore();
        }

        if (row.empty()) continue;

        y.push_back(row.back());  // Last column is target
        row.pop_back();            // Remove target from features
        X.push_back(row);          // Store feature row
    }
}

int main() {
    std::vector<std::vector<double>> X;
    std::vector<double> y;

    load_csv("../datasets/15k_hvs.csv", X, y);

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
