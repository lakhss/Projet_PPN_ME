#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include "decision_tree.hpp"
#include "../../src/DataLoader.hpp"

/**
 * @brief Main program to load CSV, build decision tree, and make predictions.
 * 
 * @return int Exit status.
 */
int main() {
    std::vector<std::vector<double>> X; ///< feature matrix
    std::vector<double> y;              ///< target vector


    // Load CSV dataset
    DataLoader::load_csv("../../datasets/15k_ga_adaptative.csv", X, y);


    // Build decision tree
    Node* tree = build_tree(X, y, 0, 10,3);

    // Make predictions for each row
    std::cout << "\n--- PREDICTIONS ---\n";
    for (size_t i = 0; i < X.size(); i++) {
        double p = predict(tree, X[i]);
        std::cout << "Row " << i+1
                  << " -> prediction = " << p
                  << " | actual = " << y[i] << "\n";
    }

    return 0;
}
