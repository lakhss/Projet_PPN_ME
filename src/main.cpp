#include "DataLoader.hpp"
#include "DecisionTreeRegressor.hpp"
#include "PerformanceEvaluator.hpp"
#include "Matrix.hpp"
#include "HistogramTreeRegressor.hpp"
#include "QuantizedDataset.hpp"

#include <vector>
#include <iostream>
#include <string>
#include <fstream>
#include <filesystem>

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

    int choice = 8; // Force le mode 8 par défaut pour nos tests HPC
    if (argc >= 2) choice = std::stoi(argv[1]);

    if(choice == 8) {
        std::cout << "=== MODE 8 : BENCHMARK COMBINED DATASET (HPC) ===\n" << std::endl;
        std::string name = "Combined";
        std::string path = resolve_dataset_path("../datasets/combined.csv");

        double outlier_threshold = 26.0;
        Matrix X;
        std::vector<double> y;
        
        std::cout << "Chargement de " << path << "..." << std::endl;
        DataLoader::load_csv(path, X, y, outlier_threshold);

        if (X.empty()) {
            std::cerr << "Erreur de chargement." << std::endl;
            return 1;
        }

        std::cout << "[HPC] Conversion AoS vers SoA..." << std::endl;
        QuantizedDataset q_dataset;
        q_dataset.ingest_from_matrix(X, y);

        std::cout << "[HPC] Préchauffage CPU..." << std::endl;
        {
            HistogramTreeRegressor dummy;
            dummy.max_depth = 5;
            dummy.fit(q_dataset);
        }

        PerformanceEvaluator::print_header();
        std::vector<int> depths = {5, 8, 10, 15};

        for (int d : depths) {
            PerformanceEvaluator::evaluate(name, "Arbre Naïf (D=" + std::to_string(d) + ")", [d]() {
                DecisionTreeRegressor t;
                t.max_depth = d;
                t.min_samples_split = 10;
                return t;
            }, X, y); 
        }

        for (int d : depths) {
            PerformanceEvaluator::evaluate_hpc(name, "HistTree HPC (D=" + std::to_string(d) + ")", [d]() {
                HistogramTreeRegressor t;
                t.max_depth = d;
                t.min_samples_split = 10;
                t.n_bins = 256; 
                return t;
            }, X, y); 
        }  
        std::cout << "\nTerminé !" << std::endl;
    } else {
        std::cout << "Mode non supporté pour le moment." << std::endl;
    }
    return 0;
}