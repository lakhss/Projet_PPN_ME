#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include "DataLoader.hpp"
#include "DecisionTreeRegressor.hpp"

void train_test_split(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& y,
    double test_ratio,
    std::vector<std::vector<double>>& X_train,
    std::vector<std::vector<double>>& X_test,
    std::vector<double>& y_train,
    std::vector<double>& y_test
) {
    size_t n = X.size();
    size_t n_test = static_cast<size_t>(n * test_ratio);

    // indices mélangés
    std::vector<int> indices(n);
    for (size_t i = 0; i < n; i++) indices[i] = i;
    std::random_shuffle(indices.begin(), indices.end());

    for (size_t i = 0; i < n; i++) {
        if (i < n_test) {
            X_test.push_back(X[indices[i]]);
            y_test.push_back(y[indices[i]]);
        } else {
            X_train.push_back(X[indices[i]]);
            y_train.push_back(y[indices[i]]);
        }
    }
}

int main() {
    std::vector<std::vector<double>> X;
    std::vector<double> y;

    // === 1. Lire ton CSV ===
    DataLoader::load_csv("15k_ga_adaptive.csv", X, y);

    std::cout << "CSV chargé : " << X.size() << " lignes, " << X[0].size()
              << " features." << std::endl;

    // === 2. Split train/test ===
    std::vector<std::vector<double>> X_train, X_test;
    std::vector<double> y_train, y_test;

    train_test_split(X, y, 0.2, X_train, X_test, y_train, y_test);

    std::cout << "Train = " << X_train.size()
              << " | Test = " << X_test.size() << std::endl;

    // === 3. Entraîner l’arbre ===
    DecisionTreeRegressor tree;
    tree.max_depth = 12;
    tree.min_samples_split = 10;

    auto start = std::chrono::high_resolution_clock::now();
    tree.fit(X_train, y_train);
    auto end = std::chrono::high_resolution_clock::now();

    double train_time = std::chrono::duration<double>(end - start).count();

    // === 4. MSE sur test ===
    double mse = 0.0;
    for (size_t i = 0; i < X_test.size(); i++) {
        double pred = tree.predict(X_test[i]);
        double err = pred - y_test[i];
        mse += err * err;
    }
    mse /= X_test.size();

    std::cout << "MSE sur test = " << mse << std::endl;
    std::cout << "Temps d'entraînement : " << train_time << " s\n";

    return 0;
}
