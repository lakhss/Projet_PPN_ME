#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include "decision_tree.cpp"   // on inclut ton fichier directement

// Chargement CSV sans dépendances
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

        // Toutes les colonnes sauf la dernière → features
        while (ss >> value) {
            row.push_back(value);
            if (ss.peek() == ',') ss.ignore();
        }

        if (row.empty()) continue;

        y.push_back(row.back());       // dernière colonne = target
        row.pop_back();                // on retire le target
        X.push_back(row);
    }
}

int main() {
    std::vector<std::vector<double>> X;
    std::vector<double> y;

    // 1. Lecture du CSV
    load_csv("../datasets/15k_ga_adaptive.csv", X, y);

    std::cout << "CSV lu : " << X.size() << " lignes, "
              << X[0].size() << " features." << std::endl;

    // 2. Construction de l'arbre
    Node* tree = build_tree(X, y);

    // 3. Prédiction de toutes les lignes
    std::cout << "\n--- PREDICTIONS ---\n";

    for (size_t i = 0; i < X.size(); i++) {
        double p = predict(tree, X[i]);
        std::cout << "Ligne " << i+1
                  << " -> prédiction = " << p
                  << " | réelle = " << y[i] << "\n";
    }

    return 0;
}
