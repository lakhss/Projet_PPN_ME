#include "DataLoader.hpp"
#include "DecisionTreeRegressor.hpp"
#include "BaggingRegressor.hpp" 
#include "BoostingRegressor.hpp"
#include "Metrics.hpp"
#include "CrossValidation.hpp"

#include <vector>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <random>
#include <iomanip> 

void get_data_from_indices(const std::vector<std::vector<double>>& X_full,
                           const std::vector<double>& y_full,
                           const std::vector<size_t>& indices,
                           std::vector<std::vector<double>>& X_out,
                           std::vector<double>& y_out) {
    X_out.clear(); y_out.clear();
    X_out.reserve(indices.size());
    y_out.reserve(indices.size());
    for (size_t i : indices) {
        X_out.push_back(X_full[i]);
        y_out.push_back(y_full[i]);
    }
}


int main(int argc, char** argv) {

    std::string dataset_path;
    int choice = 5; 
    if (argc >= 2) {
        choice = std::stoi(argv[1]);
    }

    if (choice == 6) {
        // Mode Benchmark force le dataset 15k pour l'exemple (ou 30k a changer si besoin)
        dataset_path = "../datasets/15k_ga_adaptive.csv"; 
    } else {
        switch (choice) {
            case 1: dataset_path = "../datasets/15k_ga_adaptive.csv"; break;
            case 2: dataset_path = "../datasets/15k_hvs.csv"; break;
            case 3: dataset_path = "../datasets/15k_random.csv"; break;
            case 4: dataset_path = "../datasets/30k_ga_adaptive.csv"; break;
            case 5: dataset_path = "../datasets/data_test.csv"; break;    
            default:
                std::cerr << "Choix invalide.\nUsage : ./test_loader [1-6]\n"
                          << "6 = Mode Étude Hyper-paramètres & Cross-Validation\n";
                return 1;
        }
    }

    std::cout << "Dataset sélectionné : " << dataset_path << std::endl;
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    DataLoader::load_csv(dataset_path, X, y);

    if (X.empty()) {
        std::cerr << "Erreur : Aucune donnée chargée." << std::endl;
        return 1;
    }
    if (choice == 6) {
        std::cout << "\n=== DÉBUT DE L'ÉTUDE DE PERFORMANCE (K-FOLD = 5) ===\n" << std::endl;

    int k_folds = 5;
    auto folds = CrossValidation::k_fold_split(X.size(), k_folds, 42);

    // --- ETUDE 1 : Impact de la PROFONDEUR (Max Depth) sur un Arbre Unique ---
    std::cout << "--- 1. Impact de la Profondeur (Decision Tree) ---\n";
    std::cout << std::left << std::setw(10) << "Depth" 
              << std::setw(15) << "RMSE Moyen" 
              << std::setw(15) << "Temps Moyen(s)" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    std::vector<int> depths_to_test = {5, 10, 15, 20};

    for (int depth : depths_to_test) {
        double total_rmse = 0.0;
        double total_time = 0.0;

        for (int i = 0; i < k_folds; ++i) {
            // Préparation des données du fold
            std::vector<std::vector<double>> X_train_f, X_test_f;
            std::vector<double> y_train_f, y_test_f;
            
            get_data_from_indices(X, y, folds[i].train_idx, X_train_f, y_train_f);
            get_data_from_indices(X, y, folds[i].test_idx, X_test_f, y_test_f);

            // Entraînement
            DecisionTreeRegressor dt;
            dt.max_depth = depth;
            dt.min_samples_split = 10;

            auto start = std::chrono::high_resolution_clock::now();
            dt.fit(X_train_f, y_train_f);
            auto end = std::chrono::high_resolution_clock::now();
            
            // Évaluation
            std::vector<double> preds;
            for(const auto& r : X_test_f) preds.push_back(dt.predict(r));
            
            total_rmse += Metrics::root_mean_squared_error(y_test_f, preds);
            total_time += std::chrono::duration<double>(end - start).count();
        }

        std::cout << std::left << std::setw(10) << depth 
                  << std::setw(15) << (total_rmse / k_folds) 
                  << std::setw(15) << (total_time / k_folds) << std::endl;
    }


    // --- ETUDE 2 : Impact du NOMBRE D'ARBRES sur le BOOSTING ---
    std::cout << "\n--- 2. Impact du Nombre d'Estimateurs (Boosting) ---\n";
    std::cout << "Note : Profondeur fixée à 4 (Weak Learners)\n";
    std::cout << std::left << std::setw(10) << "N_Trees" 
              << std::setw(15) << "RMSE Moyen" 
              << std::setw(15) << "Temps Moyen(s)" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    std::vector<int> estimators_to_test = {10, 30, 50};

    for (int n_est : estimators_to_test) {
        double total_rmse = 0.0;
        double total_time = 0.0;

        for (int i = 0; i < k_folds; ++i) {
            std::vector<std::vector<double>> X_train_f, X_test_f;
            std::vector<double> y_train_f, y_test_f;
            get_data_from_indices(X, y, folds[i].train_idx, X_train_f, y_train_f);
            get_data_from_indices(X, y, folds[i].test_idx, X_test_f, y_test_f);

            BoostingRegressor boost;
            boost.n_estimators = n_est;
            boost.learning_rate = 0.2;
            boost.max_depth = 4;
            boost.min_samples_split = 10;

            auto start = std::chrono::high_resolution_clock::now();
            boost.fit(X_train_f, y_train_f);
            auto end = std::chrono::high_resolution_clock::now();

            std::vector<double> preds;
            for(const auto& r : X_test_f) preds.push_back(boost.predict(r));
            
            total_rmse += Metrics::root_mean_squared_error(y_test_f, preds);
            total_time += std::chrono::duration<double>(end - start).count();
        }

        std::cout << std::left << std::setw(10) << n_est 
                  << std::setw(15) << (total_rmse / k_folds) 
                  << std::setw(15) << (total_time / k_folds) << std::endl;
    }

    std::cout << "\nFin de l'évaluation." << std::endl;

    return 0;
    }
    // Analyse rapide des données
    double min_y = *std::min_element(y.begin(), y.end());
    double max_y = *std::max_element(y.begin(), y.end());
    int count_30 = std::count(y.begin(), y.end(), 30.0); 

    std::cout << "Données chargées : " << X.size() << " lignes." << std::endl;
    std::cout << "Target (y) : min=" << min_y 
              << " | max=" << max_y 
              << " | valeurs '30.0'=" << count_30 << std::endl;

    std::vector<size_t> idx(X.size());
    std::iota(idx.begin(), idx.end(), 0);
    
    // Graine fixe (42) pour que les tests soient reproductibles / (rd) pour aléatoire
    std::mt19937 g(42); 
    std::shuffle(idx.begin(), idx.end(), g);
    
    size_t split = static_cast<size_t>(X.size() * 0.8);
    if (split == X.size()) split--; 

    // Déclaration des vecteurs d'entraînement et test
    std::vector<std::vector<double>> X_train, X_test;
    std::vector<double> y_train, y_test;

    X_train.reserve(split);
    y_train.reserve(split);
    X_test.reserve(X.size() - split);
    y_test.reserve(X.size() - split);

    for (size_t i = 0; i < split; ++i) {
        X_train.push_back(X[idx[i]]);
        y_train.push_back(y[idx[i]]);
    }
    for (size_t i = split; i < X.size(); ++i) {
        X_test.push_back(X[idx[i]]);
        y_test.push_back(y[idx[i]]);
    }

    std::cout << "Train set : " << X_train.size() << " | Test set : " << X_test.size() << "\n" << std::endl;
    std::cout << ">>> [1/3] Entraînement Arbre Unique..." << std::endl;
    
    DecisionTreeRegressor tree;
    tree.max_depth = 12;          // Profondeur identique au Bagging pour comparer a changer si besoin
    tree.min_samples_split = 10;

    auto start_tree = std::chrono::high_resolution_clock::now();
    tree.fit(X_train, y_train);
    auto end_tree = std::chrono::high_resolution_clock::now();
    
    double time_tree = std::chrono::duration<double>(end_tree - start_tree).count();
    
    // Prédiction et Évaluation Arbre
    std::vector<double> preds_tree;
    preds_tree.reserve(X_test.size());
    for (const auto& row : X_test) {
        preds_tree.push_back(tree.predict(row));
    }
    
    double mse_tree = Metrics::mean_squared_error(y_test, preds_tree);
    double rmse_tree = Metrics::root_mean_squared_error(y_test, preds_tree);
    double mae_tree = Metrics::mean_absolute_error(y_test, preds_tree);

    std::cout << " -> Terminé en " << time_tree << " s." << std::endl;
    std::cout << " -> RMSE Arbre : " << rmse_tree << "\n" << std::endl;
    std::cout << ">>> [2/3] Entraînement Bagging (Forêt)..." << std::endl;

    BaggingRegressor forest;
    forest.n_estimators = 20;       // 20 arbres
    forest.sample_ratio = 0.8;      // Chaque arbre voit 80% des données (Bootstrap)
    forest.max_depth = 12;          
    forest.min_samples_split = 10;

    auto start_bag = std::chrono::high_resolution_clock::now();
    forest.fit(X_train, y_train);
    auto end_bag = std::chrono::high_resolution_clock::now();

    double time_bag = std::chrono::duration<double>(end_bag - start_bag).count();

    // Prédiction et Évaluation Bagging
    std::vector<double> preds_bag;
    preds_bag.reserve(X_test.size());
    for (const auto& row : X_test) {
        preds_bag.push_back(forest.predict(row));
    }

    double mse_bag = Metrics::mean_squared_error(y_test, preds_bag);
    double rmse_bag = Metrics::root_mean_squared_error(y_test, preds_bag);
    double mae_bag = Metrics::mean_absolute_error(y_test, preds_bag);

    std::cout << " -> Terminé en " << time_bag << " s." << std::endl;
    std::cout << " -> RMSE Bagging : " << rmse_bag << "\n" << std::endl;
    std::cout << ">>> [3/3] Entraînement Boosting..." << std::endl;

    BoostingRegressor boost;
    boost.n_estimators = 50;     // Plus d'arbres pour le boosting
    boost.learning_rate = 0.2;   // Vitesse d'apprentissage
    boost.max_depth = 4;         // Le boosting préfère des arbres peu profonds (Weak learners)
    boost.min_samples_split = 10;

    auto start_boost = std::chrono::high_resolution_clock::now();
    boost.fit(X_train, y_train);
    auto end_boost = std::chrono::high_resolution_clock::now();

    double time_boost = std::chrono::duration<double>(end_boost - start_boost).count();

    // Prédiction et Évaluation Boosting
    std::vector<double> preds_boost;
    preds_boost.reserve(X_test.size());
    for (const auto& row : X_test) {
        preds_boost.push_back(boost.predict(row));
    }

    double mse_boost = Metrics::mean_squared_error(y_test, preds_boost);
    double rmse_boost = Metrics::root_mean_squared_error(y_test, preds_boost);
    double mae_boost = Metrics::mean_absolute_error(y_test, preds_boost);

    std::cout << " -> Terminé en " << time_boost << " s." << std::endl;
    std::cout << " -> RMSE Boosting : " << rmse_boost << "\n" << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;
    std::cout << "                        RÉSULTATS FINAUX                        " << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;
    std::cout << std::left << std::setw(15) << "Modele" 
              << std::setw(15) << "MSE" 
              << std::setw(15) << "RMSE" 
              << std::setw(15) << "MAE" 
              << std::setw(15) << "Temps (s)" << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;
    
    std::cout << std::left << std::setw(15) << "Arbre Unique" 
              << std::setw(15) << mse_tree 
              << std::setw(15) << rmse_tree 
              << std::setw(15) << mae_tree 
              << std::setw(15) << time_tree << std::endl;

    std::cout << std::left << std::setw(15) << "Bagging" 
              << std::setw(15) << mse_bag 
              << std::setw(15) << rmse_bag 
              << std::setw(15) << mae_bag 
              << std::setw(15) << time_bag << std::endl;

    std::cout << std::left << std::setw(15) << "Boosting" 
              << std::setw(15) << mse_boost 
              << std::setw(15) << rmse_boost 
              << std::setw(15) << mae_boost 
              << std::setw(15) << time_boost << std::endl;
    
    std::cout << "----------------------------------------------------------------" << std::endl;

    return 0;
}