#include "DecisionTreeRegressor.hpp"
#include <cassert>
#include <iostream>

int main() {
    std::vector<std::vector<double>> X = {
        {0.0}, {1.0}, {2.0}, {3.0}
    };
    std::vector<double> y = {0.0, 1.0, 2.0, 3.0};

    DecisionTreeRegressor tree;
    tree.max_depth = 3;
    tree.min_samples_split = 1;

    tree.fit(X, y);

    double pred = tree.predict({2.0});
    assert(std::abs(pred - 2.0) < 0.5);

    std::cout << "test_small_tree réussi" << std::endl;
    return 0;
}
