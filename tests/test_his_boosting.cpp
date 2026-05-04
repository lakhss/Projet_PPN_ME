#include "HistogramBoostingRegressor.hpp"
#include "Matrix.hpp"

#include <iostream>
#include <vector>
#include <cassert>

int main() {
    std::cout << "Test HistogramBoostingRegressor\n";

    // -------------------------
    // 1. Dataset simple
    // -------------------------
    Matrix X(4, 1);
    X(0,0) = 1.0;
    X(1,0) = 2.0;
    X(2,0) = 3.0;
    X(3,0) = 4.0;

    std::vector<double> y = {1.0, 2.0, 3.0, 4.0};

    HistogramBoostingRegressor model;
    model.n_estimators = 5;
    model.learning_rate = 0.1;
    model.max_depth = 3;
    model.min_samples_split = 2;
    model.n_bins = 10;

    // -------------------------
    // 2. Fit
    // -------------------------
    model.fit(X, y);

    // -------------------------
    // 3. Predict
    // -------------------------
    std::vector<double> x = {2.0};
    double pred = model.predict(x);

    std::cout << "Prediction = " << pred << std::endl;

    // -------------------------
    // 4. Tests simples
    // -------------------------
    assert(pred != 0.0);   // modèle a appris quelque chose

    std::cout << "Test OK " << std::endl;

    return 0;
}