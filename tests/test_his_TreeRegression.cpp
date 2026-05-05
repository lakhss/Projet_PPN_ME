#include "HistogramTreeRegressor.hpp"
#include "Matrix.hpp"

#include <iostream>
#include <vector>
#include <cassert>

int main() {
    std::cout << "Test HistogramTreeRegressor\n";

    // -------------------------
    // Dataset simple
    // -------------------------
    Matrix X(4, 1);
    X(0,0) = 1.0;
    X(1,0) = 2.0;
    X(2,0) = 3.0;
    X(3,0) = 4.0;

    std::vector<double> y = {1.0, 2.0, 3.0, 4.0};

    HistogramTreeRegressor model;
    model.max_depth = 3;
    model.min_samples_split = 1;
    model.n_bins = 5;
    model.min_gain = 0.0;

    // -------------------------
    // Fit
    // -------------------------
    model.fit(X, y);

    // -------------------------
    // Predict
    // -------------------------
    double p1 = model.predict({2.0});
    double p2 = model.predict({2.0});

    std::cout << "Prediction = " << p1 << std::endl;

    // -------------------------
    // Tests
    // -------------------------
    assert(p1 == p2);     // stabilité
    assert(p1 != 0.0);    // apprentissage

    std::cout << "Test OK " << std::endl;

    return 0;
}