#include "DataLoader.hpp"
#include "DecisionTreeRegressor.hpp"
#include "BaggingRegressor.hpp" 
#include "BoostingRegressor.hpp"
#include "PerformanceEvaluator.hpp"

#include <vector>
#include <iostream>
#include <string>
#include <fstream>
#include <map>

int main(int argc, char** argv) {

    std::ofstream outfile("results.csv");
    if (outfile.is_open()) {
        outfile << "Dataset,Model,RMSE,MAE,Time\n"; 
        outfile.close();
    }

    int choice = 5; 
    if (argc >= 2) choice = std::stoi(argv[1]);

    std::map<std::string, std::string> all_datasets = {
        {"Adaptive", "../datasets/15k_ga_adaptive.csv"},
        {"HVS",      "../datasets/15k_hvs.csv"},
        {"Random",   "../datasets/15k_random.csv"}
        
    };

   
    if (choice == 6) {
        std::cout << "=== MODE BENCHMARK MULTI-DATASETS ===\n" << std::endl;

        for (const auto& [name, path] : all_datasets) {
            std::cout << "\n################################################" << std::endl;
            std::cout << " CHARGEMENT : " << name << " (" << path << ")" << std::endl;
            std::cout << "################################################" << std::endl;

            std::vector<std::vector<double>> X;
            std::vector<double> y;
            DataLoader::load_csv(path, X, y);

            if (X.empty()) {
                std::cerr << "-> Erreur ou Fichier vide. Je passe au suivant." << std::endl;
                continue;
            }

            PerformanceEvaluator::print_header();

            // A. Étude Profondeur (Arbre)
            std::vector<int> depths = {5, 10, 15, 20};
            for (int d : depths) {
                PerformanceEvaluator::evaluate(name, "Arbre (Depth=" + std::to_string(d) + ")", [d]() {
                    DecisionTreeRegressor t; t.max_depth = d; t.min_samples_split = 10; return t;
                }, X, y);
            }

            // B. Étude Boosting
            std::vector<int> n_trees = {10, 30, 50};
            for (int n : n_trees) {
                PerformanceEvaluator::evaluate(name, "Boosting (N=" + std::to_string(n) + ")", [n]() {
                    BoostingRegressor b; b.n_estimators = n; b.max_depth = 4; b.learning_rate = 0.2; return b;
                }, X, y);
            }

            // C. Étude Bagging
            for (int n : n_trees) {
                PerformanceEvaluator::evaluate(name, "Bagging (N=" + std::to_string(n) + ")", [n]() {
                    BaggingRegressor bg; bg.n_estimators = n; bg.max_depth = 12; return bg;
                }, X, y);
            }
        }
        std::cout << "\nTerminé ! Résultats dans build/results.csv" << std::endl;
    } 

    else {
        std::string path;
        std::string name = "SingleRun"; 
        switch (choice) {
            case 1: path = "../datasets/15k_ga_adaptive.csv"; name="Adaptive"; break;
            case 2: path = "../datasets/15k_hvs.csv"; name="HVS"; break;
            case 3: path = "../datasets/15k_random.csv"; name="Random"; break;
            case 4: path = "../datasets/30k_ga_adaptive.csv"; name="Adaptive30k"; break;
            case 5: path = "../datasets/data_test.csv"; name="Test"; break;
            default: std::cerr << "Choix invalide.\n"; return 1;
        }
        
        std::cout << "Dataset : " << path << std::endl;
        std::vector<std::vector<double>> X;
        std::vector<double> y;
        DataLoader::load_csv(path, X, y);
        
        PerformanceEvaluator::print_header();
        
        PerformanceEvaluator::evaluate(name, "Arbre (D=12)", [](){ 
            DecisionTreeRegressor t; t.max_depth=12; t.min_samples_split=10; return t; 
        }, X, y);
        
        PerformanceEvaluator::evaluate(name, "Bagging (N=20)", [](){ 
            BaggingRegressor b; b.n_estimators=20; b.max_depth=12; return b; 
        }, X, y);

        PerformanceEvaluator::evaluate(name, "Boosting (N=50)", [](){ 
            BoostingRegressor b; b.n_estimators=50; b.max_depth=4; b.learning_rate=0.2; return b; 
        }, X, y);
    }

    return 0;
}