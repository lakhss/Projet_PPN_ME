#pragma once
#include "DecisionTreeRegressor.hpp"
#include <vector>

class BoostingRegressor {
public:
    int n_estimators = 50;
    double learning_rate = 0.1; // strength of each tree contribution
    int max_depth = 3;
    int min_samples_split = 6;

    /**
     * * @brief train the Boosting Regressor on the given data
     */
    void fit(const std::vector<std::vector<double>>& X,
             const std::vector<double>& y);


    /**
     * * @brief predict the value of a sample
     */
    double predict(const std::vector<double>& x) const;

private:
    double init_value = 0.0;   // initial prediction (mean of y)
    std::vector<DecisionTreeRegressor> trees;   // ensemble of trees
};