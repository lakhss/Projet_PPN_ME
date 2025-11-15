#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include "decision_tree.cpp"   

/**
 * @brief Loads a CSV file into feature matrix X and target vector y.
 * 
 * The last column of each row is considered as the target value.
 * 
 * @param filename Path to the CSV file.
 * @param X Output feature matrix (each row is a sample, columns are features).
 * @param y Output target vector (last column of each row in the CSV).
 */
void load_csv(const std::string& filename,
              std::vector<std::vector<double>>& X,
              std::vector<double>& y)
{
    std::ifstream file(filename);
    std::string line;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<double> row;
        double value;

        // Read all values in the line, separated by commas
        while (ss >> value) {
            row.push_back(value);
            if (ss.peek() == ',') ss.ignore();
        }

        if (row.empty()) continue;

        y.push_back(row.back());  // last value is the target
        row.pop_back();            // remove target from feature row
        X.push_back(row);
    }
}

/**
 * @brief Main program to load CSV, build decision tree, and make predictions.
 * 
 * @return int Exit status.
 */
int main() {
    std::vector<std::vector<double>> X; ///< feature matrix
    std::vector<double> y;              ///< target vector


    std::vector<std::vector<double>> X1;
    std::vector<double> y1;


    // Load CSV dataset
    load_csv("../datasets/15k_ga_adaptive.csv", X, y);

    // La j'essaie avec l'autre dataset
     load_csv("../datasets/15k_hvs.csv", X1, y1);

    std::cout << "CSV read: " << X.size() << " rows, "
              << X[0].size() << " features." << std::endl;



    std::cout << "Second CSV read: " << X1.size() << " rows, "
          << X1[0].size() << " features." << std::endl;
  

    // Build decision tree
    Node* tree = build_tree(X, y);

    // Make predictions for each row
    std::cout << "\n--- PREDICTIONS ---\n";
    for (size_t i = 0; i < X.size(); i++) {
        double p = predict(tree, X[i]);
        std::cout << "Row " << i+1
                  << " -> prediction = " << p
                  << " | actual = " << y[i] << "\n";
    }

    std::cout << "\n--- PREDICTIONS ON SECOND CSV ---\n";
    for (size_t i = 0; i < X1.size(); i++) {
        double p = predict(tree, X1[i]);
        std::cout << "Row " << i+1
              << " -> prediction = " << p
              << " | actual = " << y1[i] << "\n";
}

    return 0;
}
