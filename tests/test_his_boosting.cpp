#include <gtest/gtest.h>

#include "HistogramBoostingRegressor.hpp"
#include "Matrix.hpp"

// ----------------------------
// Dataset simple
// ----------------------------
TEST(HistogramBoosting, BasicFitPredict) {
    Matrix X(4, 1);
    X(0,0) = 1.0;
    X(1,0) = 2.0;
    X(2,0) = 3.0;
    X(3,0) = 4.0;

    std::vector<double> y = {1.0, 2.0, 3.0, 4.0};

    HistogramBoostingRegressor model;
    model.n_estimators = 10;
    model.learning_rate = 0.1;
    model.max_depth = 3;
    model.min_samples_split = 2;
    model.n_bins = 16;

    model.fit(X, y);

    double pred = model.predict({2.0});

    EXPECT_NE(pred, 0.0);  
}

// ----------------------------
// Test dataset vide
// ----------------------------
TEST(HistogramBoosting, EmptyDataset) {
    Matrix X;
    std::vector<double> y;

    HistogramBoostingRegressor model;

    model.fit(X, y);

    double pred = model.predict({1.0});

    EXPECT_EQ(pred, 0.0);
}

// ---------------
// Test stabilité 
// ---------------
TEST(HistogramBoosting, Deterministic) {
    Matrix X(3, 1);
    X(0,0) = 1.0;
    X(1,0) = 2.0;
    X(2,0) = 3.0;

    std::vector<double> y = {1.0, 2.0, 3.0};

    HistogramBoostingRegressor model;
    model.n_estimators = 10;
    model.learning_rate = 0.1;
    model.max_depth = 3;
    model.min_samples_split = 2;
    model.n_bins = 16;

    model.fit(X, y);

    double p1 = model.predict({2.0});
    double p2 = model.predict({2.0});

    EXPECT_NEAR(p1, p2, 1e-9);  
}