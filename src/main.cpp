#include "DataLoader.hpp"
#include "DecisionTreeRegressor.hpp"
#include "BaggingRegressor.hpp"
#include "BoostingRegressor.hpp"
#include "PerformanceEvaluator.hpp"
#include "Matrix.hpp"
#include "HistogramTreeRegressor.hpp"

#include <vector>
#include <iostream>
#include <string>
#include <fstream>
#include <map>
#include <filesystem>
//#include <cstddef>   utilisé dans le bloc de vérification mémoire 

static std::string resolve_dataset_path(const std::string& relative_path) {
    namespace fs = std::filesystem;
    const std::vector<fs::path> candidates = {
        fs::path(relative_path),
        fs::path("../") / relative_path,
        fs::path("../../") / relative_path,
        fs::path("datasets") / fs::path(relative_path).filename()
    };

    for (const auto& p : candidates) {
        if (fs::exists(p)) return p.string();
    }
    return relative_path;
}

int main(int argc, char** argv) {
    std::ofstream outfile("results.csv");
    if (outfile.is_open()) {
        outfile << "Dataset,Model,RMSE,MAE,MAPE,Time\n";
        outfile.close();
    }

    int choice = 5;
    if (argc >= 2) choice = std::stoi(argv[1]);

    std::map<std::string, std::string> all_datasets = {
        {"Adaptive", "../datasets/15k_ga_adaptive.csv"},
        {"HVS", "../datasets/15k_hvs.csv"},
        {"Random", "../datasets/15k_random.csv"},
        {"Adaptive30k", "../datasets/30k_ga_adaptive.csv"}
    };

    if (choice == 6) {
        std::cout << "=== MODE BENCHMARK MULTI-DATASETS ===\n" << std::endl;

        for (const auto& [name, path] : all_datasets) {
            std::cout << "\n################################################" << std::endl;
            std::cout << " CHARGEMENT : " << name << " (" << path << ")" << std::endl;
            std::cout << "################################################" << std::endl;

            double outlier_threshold = 26.0;

            Matrix X;
            std::vector<double> y;
            DataLoader::load_csv(resolve_dataset_path(path), X, y, outlier_threshold);

            if (X.empty()) {
                std::cerr << "-> Erreur ou Fichier vide. Je passe au suivant." << std::endl;
                continue;
            }

            PerformanceEvaluator::print_header();

            std::vector<int> depths = {5, 8, 10, 15};
            for (int d : depths) {
                PerformanceEvaluator::evaluate(name, "Arbre (Depth=" + std::to_string(d) + ")", [d]() {
                    DecisionTreeRegressor t;
                    t.max_depth = d;
                    t.min_samples_split = 10;
                    return t;
                }, X, y);
            }

            // Etude de l'arbre histogramme
            for (int d : depths) {
                PerformanceEvaluator::evaluate(name, "HistTree (D=" + std::to_string(d) + ", B=16)", [d]() {
                    HistogramTreeRegressor t;
                    t.max_depth = d;
                    t.min_samples_split = 10;
                    t.n_bins = 16;
                    return t;
                }, X, y);
            }  

            std::vector<int> n_trees = {10, 30, 50};
            for (int n : n_trees) {
                PerformanceEvaluator::evaluate(name, "Boosting (N=" + std::to_string(n) + ")", [n]() {
                    BoostingRegressor b;
                    b.n_estimators = n;
                    b.max_depth = 4;
                    b.learning_rate = 0.2;
                    return b;
                }, X, y);
            }

            for (int n : n_trees) {
                PerformanceEvaluator::evaluate(name, "Bagging (N=" + std::to_string(n) + ")", [n]() {
                    BaggingRegressor bg;
                    bg.n_estimators = n;
                    bg.max_depth = 12;
                    return bg;
                }, X, y);
            }
        }

        std::cout << "\nTerminé ! Résultats dans results.csv" << std::endl;
    }
    else if (choice == 7) {
        std::cout << "=== MODE 7 : ETUDE DE CONVERGENCE (Parametric Study) ===\n" << std::endl;

        std::string dataset_path = "../datasets/15k_ga_adaptive.csv";
        std::string dataset_name = "Adaptive30k_Study";

        Matrix X;
        std::vector<double> y;
        DataLoader::load_csv(resolve_dataset_path(dataset_path), X, y, 26.0);

        if (X.empty()) return 1;

        PerformanceEvaluator::print_header();

        std::vector<int> n_list = {1, 5, 10, 20, 30, 50, 75, 100};

        std::cout << "\n--- ETUDE BAGGING (Depth=12) ---\n";
        for (int n : n_list) {
            std::string name = "Bag_D12_N" + std::to_string(n);
            PerformanceEvaluator::evaluate(dataset_name, name, [n]() {
                BaggingRegressor b;
                b.n_estimators = n;
                b.max_depth = 12;
                return b;
            }, X, y);
        }

        std::cout << "\n--- ETUDE BOOSTING WEAK (Depth=4) ---\n";
        for (int n : n_list) {
            std::string name = "Boost_D4_N" + std::to_string(n);
            PerformanceEvaluator::evaluate(dataset_name, name, [n]() {
                BoostingRegressor b;
                b.n_estimators = n;
                b.max_depth = 4;
                b.learning_rate = 0.2;
                return b;
            }, X, y);
        }

        std::cout << "\n--- ETUDE BOOSTING MEDIUM (Depth=8) ---\n";
        for (int n : n_list) {
            std::string name = "Boost_D8_N" + std::to_string(n);
            PerformanceEvaluator::evaluate(dataset_name, name, [n]() {
                BoostingRegressor b;
                b.n_estimators = n;
                b.max_depth = 8;
                b.learning_rate = 0.2;
                return b;
            }, X, y);
        }
    }
    else if(choice == 8){
        std::cout << "=== MODE 8 : BENCHMARK COMBINED DATASET ===\n" << std::endl;

        auto name ="Combined";
        auto path ="../datasets/combined.csv";

        std::cout << "\n################################################" << std::endl;
            std::cout << " CHARGEMENT : " << name << " (" << path << ")" << std::endl;
            std::cout << "################################################" << std::endl;

            double outlier_threshold = 26.0;

            Matrix X;
            std::vector<double> y;
            DataLoader::load_csv(resolve_dataset_path(path), X, y, outlier_threshold);

            if (X.empty()) {
                std::cerr << "-> Erreur ou Fichier vide. Je passe au suivant." << std::endl;
            }

            PerformanceEvaluator::print_header();

            std::vector<int> depths = {5, 8, 10, 15};
            for (int d : depths) {
                PerformanceEvaluator::evaluate(name, "Arbre (Depth=" + std::to_string(d) + ")", [d]() {
                    DecisionTreeRegressor t;
                    t.max_depth = d;
                    t.min_samples_split = 10;
                    return t;
                }, X, y);
            }

            // Etude de l'arbre histogramme
            for (int d : depths) {
                PerformanceEvaluator::evaluate(name, "HistTree (D=" + std::to_string(d) + ", B=16)", [d]() {
                    HistogramTreeRegressor t;
                    t.max_depth = d;
                    t.min_samples_split = 10;
                    t.n_bins = 16;
                    return t;
                }, X, y);
            }  

            std::vector<int> n_trees = {10, 30, 50};
            for (int n : n_trees) {
                PerformanceEvaluator::evaluate(name, "Boosting (N=" + std::to_string(n) + ")", [n]() {
                    BoostingRegressor b;
                    b.n_estimators = n;
                    b.max_depth = 4;
                    b.learning_rate = 0.2;
                    return b;
                }, X, y);
            }

            for (int n : n_trees) {
                PerformanceEvaluator::evaluate(name, "Bagging (N=" + std::to_string(n) + ")", [n]() {
                    BaggingRegressor bg;
                    bg.n_estimators = n;
                    bg.max_depth = 12;
                    return bg;
                }, X, y);
            }

            std::cout << "\nTerminé ! Résultats dans results.csv" << std::endl;
        }

    else {
        std::string path;
        std::string name = "SingleRun";

        switch (choice) {
            case 1: path = "../datasets/15k_ga_adaptive.csv"; name = "Adaptive"; break;
            case 2: path = "../datasets/15k_hvs.csv"; name = "HVS"; break;
            case 3: path = "../datasets/15k_random.csv"; name = "Random"; break;
            case 4: path = "../datasets/30k_ga_adaptive.csv"; name = "Adaptive30k"; break;
            case 5: path = "../datasets/data_test.csv"; name = "Test"; break;
            case 6: path = "../datasets/combined.csv"; name = "Combined"; break;
            default:
                std::cerr << "Choix invalide.\n";
                return 1;
        }

        std::cout << "Dataset : " << path << std::endl;

        Matrix X;
        std::vector<double> y;
        path = resolve_dataset_path(path);
        DataLoader::load_csv(path, X, y);

        if (X.empty()) {
            std::cerr << "Dataset vide ou invalide.\n";
            return 1;
        }

        /*  verification memoire et stockage contigu des lignes
        if (X.rows() >= 2 && X.cols() > 0) {
            std::cout << "\n=== Verification memoire ===\n";
            std::cout << "Adresse debut ligne 0 : " << (const void*)X.row_data(0) << std::endl;
            std::cout << "Adresse debut ligne 1 : " << (const void*)X.row_data(1) << std::endl;

            std::ptrdiff_t diff = X.row_data(1) - X.row_data(0);
            std::cout << "Ecart (en doubles) : " << diff << std::endl;
            std::cout << "Nb colonnes ligne 0 : " << X.cols() << std::endl;

            if (diff == static_cast<std::ptrdiff_t>(X.cols())) {
                std::cout << "=> lignes adjacentes en memoire (stockage contigu)\n";
            } else {
                std::cout << "=> stockage NON contigu\n";
            }
        }*/

        PerformanceEvaluator::print_header();

        PerformanceEvaluator::evaluate(name, "Arbre (D=12)", []() {
            DecisionTreeRegressor t;
            t.max_depth = 12;
            t.min_samples_split = 10;
            return t;
        }, X, y);



          // Etude de l'arbre histogramme avec plusieurs nombres de bins
        /*PerformanceEvaluator::evaluate(name, "HistTree (D=12, B=8)", []() {
            HistogramTreeRegressor t;
            t.max_depth = 12;
            t.min_samples_split = 10;
            t.n_bins = 8;
            return t;
        }, X, y);

        PerformanceEvaluator::evaluate(name, "HistTree (D=12, B=16)", []() {
            HistogramTreeRegressor t;
            t.max_depth = 12;
            t.min_samples_split = 10;
            t.n_bins = 16;
            return t;
        }, X, y);*/

        PerformanceEvaluator::evaluate(name, "HistTree (D=12, B=32)", []() {
            HistogramTreeRegressor t;
            t.max_depth = 12;
            t.min_samples_split = 10;
            t.n_bins = 32;
            return t;
        }, X, y);


        PerformanceEvaluator::evaluate(name, "Bagging (N=20)", []() {
            BaggingRegressor b;
            b.n_estimators = 20;
            b.max_depth = 12;
            return b;
        }, X, y);

        PerformanceEvaluator::evaluate(name, "Boosting (N=50)", []() {
            BoostingRegressor b;
            b.n_estimators = 50;
            b.max_depth = 4;
            b.learning_rate = 0.2;
            return b;
        }, X, y);
    }

    return 0;
}