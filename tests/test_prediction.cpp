//S’assurer que predict() ne plante pas et retourne une valeur valide

#include "DecisionTreeRegressor.hpp"
#include <cassert>
#include <iostream>
#include <cmath>

int main() {
    std::vector<std::vector<double>> X = {
        {0.1, 0.2},
        {0.2, 0.3},
        {0.3, 0.4},
        {0.4, 0.5}
    };
    std::vector<double> y = {1.0, 1.5, 2.0, 2.5};

    DecisionTreeRegressor tree;
    tree.fit(X, y);

    double pred = tree.predict({0.25, 0.35});

    assert(!std::isnan(pred));
    assert(pred > 0.0);

    std::cout << "test_prediction réussi" << std::endl;
    return 0;
}
