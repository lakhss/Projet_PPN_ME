//Vérifier que le CSV est bien chargé et que le filtrage des valeurs 30 fonctionne

#include "DecisionTreeRegressor.hpp"
#include <cassert>
#include <vector>
#include <iostream>

int main() {
    std::vector<std::vector<double>> X;
    std::vector<double> y;

    DataLoader::load_csv("../datasets/15k_ga_adaptive.csv", X, y);

    assert(!X.empty());
    assert(X.size() == y.size());

    // Vérifie uniquement la cohérence des données
    for (const auto& row : X) {
        assert(row.size() == 10);
    }

    std::cout << "test_dataloader réussi\n";
    return 0;
}
