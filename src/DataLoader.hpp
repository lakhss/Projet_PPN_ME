#pragma once

#include "Matrix.hpp"

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

class DataLoader {
public:
    static void load_csv(const std::string& filename,
                         Matrix& X,
                         std::vector<double>& y,
                         double max_perf_threshold = 26.0,
                         size_t reserve_size = 35000)
    {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "ERREUR : Impossible d'ouvrir " << filename << std::endl;
            X.resize(0, 0);
            y.clear();
            return;
        }

        std::vector<double> flat_data;
        flat_data.reserve(reserve_size * 10);
        y.clear();
        y.reserve(reserve_size);

        std::string line;
        std::getline(file, line); // skip header

        int total_lines = 0;
        int kept_lines = 0;
        constexpr size_t n_features = 10;

        while (std::getline(file, line)) {
            total_lines++;
            std::stringstream ss(line);
            std::vector<double> row;
            row.reserve(11);

            double val;
            while (ss >> val) {
                row.push_back(val);
                if (ss.peek() == ',') ss.ignore();
            }

            if (row.size() != 11) {
                continue;
            }

            const double perf_val = row.back();
            if (perf_val > max_perf_threshold) {
                continue;
            }

            row.pop_back(); // remove target

            for (double x : row) {
                flat_data.push_back(x);
            }

            y.push_back(perf_val);
            kept_lines++;
        }

        X.resize(kept_lines, n_features);
        for (size_t i = 0; i < flat_data.size(); ++i) {
            X(i / n_features, i % n_features) = flat_data[i];
        }

        std::cout << "[DataLoader] " << filename
                  << " : " << kept_lines << " lignes chargees"
                  << " (Filtres: " << (total_lines - kept_lines)
                  << ", Seuil=" << max_perf_threshold << ")"
                  << std::endl;
    }
};