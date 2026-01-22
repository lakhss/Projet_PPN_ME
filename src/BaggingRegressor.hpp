#pragma once
#include "DecisionTreeRegressor.hpp"
#include <vector>
#include <random>  

class BaggingRegressor {
public:
    int n_estimators = 15;          // number of trees 
    double sample_ratio = 1.0;      // 1.0 => bootstrap size = dataset size (N)
    int max_depth = 10;
    int min_samples_split = 6;      
    unsigned int seed = 42;

    /**
     * @brief Train the Bagging regression model.
     */
    void fit(const std::vector<std::vector<double>>& X,   
             const std::vector<double>& y);

    /**
     * @brief Predict the target value for a given input sample.
     */
    double predict(const std::vector<double>& x) const;

private:
    std::vector<DecisionTreeRegressor> trees;
};