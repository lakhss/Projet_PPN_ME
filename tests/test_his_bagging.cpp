#include <gtest/gtest.h>

#include "HistogramBaggingRegressor.hpp"
#include "Matrix.hpp"
#include <cmath>

// -----------------------------
// 1. Dataset vide
// -----------------------------
TEST(HistogramBagging, EmptyDataset) {
    HistogramBaggingRegressor model;

    Matrix X;
    std::vector<double> y;

    model.fit(X, y);

    double pred = model.predict({1.0, 2.0});

    EXPECT_DOUBLE_EQ(pred, 0.0);
}

// -----------------------------
// 2. Apprentissage simple
// -----------------------------
TEST(HistogramBagging, SimpleLearning) {
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

    std::cout << "Prediction = " << pred << std::endl;

    EXPECT_GT(pred, 0.0);
}

// -----------------------------
// 3. Stabilité bagging
// -----------------------------
TEST(HistogramBagging, Stability) {
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

    EXPECT_NEAR(p1, p2, 1e-9);
}