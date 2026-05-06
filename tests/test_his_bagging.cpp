#include <gtest/gtest.h>

#include "HistogramBaggingRegressor.hpp"
#include "QuantizedDataset.hpp"

// ----------------------
// Dataset simple
// ----------------------
TEST(HistogramBaggingQuantized, BasicFitPredict) {
    // Dataset simple
    Matrix X(4, 1);
    X(0,0) = 1.0;
    X(1,0) = 2.0;
    X(2,0) = 3.0;
    X(3,0) = 4.0;

    std::vector<double> y = {1.0, 2.0, 3.0, 4.0};

    // création dataset quantifié
    QuantizedDataset dataset(X, y, 16); // 16 bins

    HistogramBaggingRegressor model;
    model.n_estimators = 10;
    model.sample_ratio = 0.8;
    model.seed = 42;

    model.fit(dataset);

    double pred = model.predict({2.0});

    EXPECT_NE(pred, 0.0);  // le modèle apprend
}

// ----------------------
// Test stabilité
// ----------------------
TEST(HistogramBaggingQuantized, Deterministic) {
    Matrix X(3, 1);
    X(0,0) = 1.0;
    X(1,0) = 2.0;
    X(2,0) = 3.0;

    std::vector<double> y = {1.0, 2.0, 3.0};

    QuantizedDataset dataset(X, y, 16);

    HistogramBaggingRegressor model;
    model.n_estimators = 10;
    model.sample_ratio = 1.0;
    model.seed = 123;

    model.fit(dataset);

    double p1 = model.predict({2.0});
    double p2 = model.predict({2.0});

    EXPECT_NEAR(p1, p2, 1e-9);  // stable
}

// ----------------------
// Dataset vide
// ----------------------
TEST(HistogramBaggingQuantized, EmptyDataset) {
    Matrix X;
    std::vector<double> y;

    QuantizedDataset dataset(X, y, 16);

    HistogramBaggingRegressor model;
    model.fit(dataset);

    double pred = model.predict({1.0});

    EXPECT_EQ(pred, 0.0);
}