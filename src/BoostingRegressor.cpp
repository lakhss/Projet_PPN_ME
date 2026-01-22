#include "BoostingRegressor.hpp"
#include <numeric>


/**
 * * @brief train the Boosting Regressor on the given data
 *   
 * * The Boosting Regressor builds an ensemble of Decision Trees
 * * Each tree is trained on the residuals of the previous trees' predictions
 * * @param X Feature matrix
 * * @param y Target values
 * 
 */
void BoostingRegressor::fit(const std::vector<std::vector<double>>& X,
                            const std::vector<double>& y) {

    trees.clear();     
    trees.reserve(n_estimators);

  
    //Initialization: constant prediction = mean of y
    init_value = std::accumulate(y.begin(), y.end(), 0.0) / y.size();  

    std::vector<double> preds(y.size(), init_value);
    std::vector<double> residuals(y.size());  // to store residuals

    for (int m = 0; m < n_estimators; ++m) {
        for (size_t i = 0; i < y.size(); ++i)
            residuals[i] = y[i] - preds[i]; // residual = true value - current prediction

        // Train a new Decision Tree on the residuals
        DecisionTreeRegressor tree;
        tree.max_depth = max_depth;
        tree.min_samples_split = min_samples_split;
        tree.fit(X, residuals);

        // Update predictions
        for (size_t i = 0; i < X.size(); ++i)
            preds[i] += learning_rate * tree.predict(X[i]);

        trees.push_back(std::move(tree));
    }
}
/**
 * * @brief predict the value of a sample
 * 
 * * The prediction is the sum of the initial prediction and
 * * the contributions from each tree in the ensemble
 * * @param x Input sample
 * * @return Predicted value
 */
double BoostingRegressor::predict(const std::vector<double>& x) const {
    double y_pred = init_value;
    for (const auto& t : trees)  // iterate through each tree
        y_pred += learning_rate * t.predict(x);
    return y_pred;
}