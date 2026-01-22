#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

class DataLoader {
public:
    static void load_csv(const std::string& filename,
                         std::vector<std::vector<double>>& X,
                         std::vector<double>& y,
                         double max_perf_threshold = 26.0, 
                         size_t reserve_size = 35000)      
    {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "ERREUR : Impossible d'ouvrir " << filename << std::endl;
            return;
        }

        // Nettoyage et réservation mémoire 
        y.clear();
        X.reserve(reserve_size);
        y.reserve(reserve_size);

        std::string line;
        std::getline(file, line); 

        int total_lines = 0;
        int kept_lines = 0;

        while (std::getline(file, line)) {
            total_lines++;
            std::stringstream ss(line);
            std::vector<double> row;
            double val;

            while (ss >> val) {
                row.push_back(val);
                if (ss.peek() == ',') ss.ignore();
            }

            if (row.size() != 11) {
                continue; 
            }

            double perf_val = row.back();

            // FILTRAGE : Si la perf dépasse le seuil, on ignore la ligne
            if (perf_val > max_perf_threshold) {
                continue; 
            }

            y.push_back(perf_val);
            row.pop_back(); 
            X.push_back(row);
            kept_lines++;
        }

        std::cout << "[DataLoader] " << filename 
                  << " : " << kept_lines << " lignes chargees"
                  << " (Filtres: " << (total_lines - kept_lines) 
                  << ", Seuil=" << max_perf_threshold << ")" 
                  << std::endl;
    }
};