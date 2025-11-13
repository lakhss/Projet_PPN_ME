#include <iostream>
#include <vector>
#include <chrono>
#include "DataLoader.hpp"
#include "DecisionTreeRegressor.hpp"

int main() {
    std::vector<std::vector<double>> X, X_train, X_test;
    std::vector<double> y, y_train, y_test;

    DataLoader::load_csv("15k_ga_adaptive.csv", X, y);
      
    double min_y = *std::min_element(y.begin(), y.end());
    double max_y = *std::max_element(y.begin(), y.end());
    int count_30 = std::count(y.begin(), y.end(), 30.0);
    std::cout << "Performance : min = " << min_y 
              << " | max = " << max_y 
              << " | valeurs exactement 30.0 = " << count_30 << std::endl;

    std::random_device rd; std::mt19937 g(rd());
    std::vector<size_t> idx(X.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), g);
    size_t split = X.size() * 0.8;

    for (size_t i = 0; i < split; ++i) {
        X_train.push_back(X[idx[i]]);
        y_train.push_back(y[idx[i]]);
    }
    for (size_t i = split; i < X.size(); ++i) {
        X_test.push_back(X[idx[i]]);
        y_test.push_back(y[idx[i]]);
    }

    // Entraînement
    DecisionTreeRegressor tree;
    tree.max_depth = 12;
    tree.min_samples_split = 10;

    auto start = std::chrono::high_resolution_clock::now();
    tree.fit(X_train, y_train);
    auto end = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration<double>(end - start).count();

    // Prédictions
    double mse = 0.0;
    for (size_t i = 0; i < X_test.size(); ++i) {
        double pred = tree.predict(X_test[i]);
        mse += (pred - y_test[i]) * (pred - y_test[i]);
    }
    mse /= X_test.size();

    std::cout << "MSE sur test : " << mse << std::endl;
    std::cout << "Temps entraînement : " << time << " s" << std::endl;

    return 0;
}