#include <iostream>
#include <fstream>
#include <sstream>
#include <vector> 
#include "decision_tree.hpp"



// FONCTION : load_csv

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

        while (ss >> value) {
            row.push_back(value);
            if (ss.peek() == ',') ss.ignore();
        }

        if (row.empty()) continue;

        y.push_back(row.back());   
        X.push_back(row);
    }
}


int main() {
    std::vector<std::vector<double>> X;
    std::vector<double> y;

    //  CHARGE LE BON CSV
    load_csv("../datasets/15k_hvs.csv", X, y);

    std::cout << "Dataset chargé : " << X.size() 
              << " lignes, " << X[0].size() 
              << " features.\n";

    // CONSTRUIT L’ARBRE POUR CE DATASET
    Node* tree = build_tree(X, y);

    std::cout << "\n--- PREDICTIONS HVS ---\n";
    for (size_t i = 0; i < X.size(); i++) {
        double p = predict(tree, X[i]);
        std::cout << "Ligne " << i+1
                  << " -> prediction = " << p
                  << " | réel = " << y[i] << "\n";
    }

    return 0;
}
