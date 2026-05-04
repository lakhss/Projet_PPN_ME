#include "HistogramBaggingRegressor.hpp"
#include "Matrix.hpp"

#include <cassert>
#include <iostream>
#include <vector>

int main() {
    std::cout << "Test HistogramBaggingRegressor\n";

    // -----------------------------
    // 1. Test dataset vide
    // -----------------------------
    {
        HistogramBaggingRegressor model;

        Matrix X;
        std::vector<double> y;

        model.fit(X, y);

        double pred = model.predict({1.0, 2.0});

        assert(pred == 0.0);

        std::cout << "Test dataset vide OK\n";
    }

    // -----------------------------
    // 2. Test simple apprentissage
    // -----------------------------
    {
        Matrix X(3, 1);
        X(0, 0) = 1.0;
        X(1, 0) = 2.0;
        X(2, 0) = 3.0;

        std::vector<double> y = {1.0, 2.0, 3.0};

        HistogramBaggingRegressor model;
        model.n_estimators = 5;
        model.sample_ratio = 1.0;
        model.seed = 42;

        model.fit(X, y);

        double pred = model.predict({2.0});

        std::cout << "Prediction = " << pred << "\n";

        // juste vérifier que ça produit quelque chose de cohérent
        assert(pred > 0.0);

        std::cout << "Test apprentissage OK\n";
    }

    // -----------------------------
    // 3. Test stabilité bagging
    // -----------------------------
    {
        Matrix X(4, 1);
        X(0, 0) = 1.0;
        X(1, 0) = 2.0;
        X(2, 0) = 3.0;
        X(3, 0) = 4.0;

        std::vector<double> y = {1.0, 2.0, 3.0, 4.0};

        HistogramBaggingRegressor model;
        model.n_estimators = 10;
        model.sample_ratio = 0.8;
        model.seed = 123;

        model.fit(X, y);

        double p1 = model.predict({3.0});
        double p2 = model.predict({3.0});

        // bagging doit être stable
        assert(std::abs(p1 - p2) < 1e-9);

        std::cout << "Test stabilité OK\n";
    }

    std::cout << "Tous les tests OK ok \n";

    return 0;
}