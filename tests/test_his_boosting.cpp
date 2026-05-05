#include "HistogramBoostingRegressor.hpp"
#include "Matrix.hpp"

#include <gtest/gtest.h>
#include <vector>

TEST(HistogramBoosting, SimpleDataset) {
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

    model.fit(X, y);

    double pred = model.predict({2.0});

    EXPECT_NE(pred, 0.0);   // modèle apprend quelque chose
}

TEST(HistogramBoosting, PredictionReasonable) {
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
    model.n_bins = 10;

    model.fit(X, y);

    double pred = model.predict({2.0});

    // valeur attendue autour de 2
    EXPECT_GT(pred, 1.0);
    EXPECT_LT(pred, 3.0);
}