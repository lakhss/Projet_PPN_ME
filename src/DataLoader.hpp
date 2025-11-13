#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

class DataLoader {
public:
    static void load_csv(const std::string& filename,
                         std::vector<std::vector<double>>& X,
                         std::vector<double>& y) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "ERREUR : Impossible d'ouvrir " << filename << std::endl;
            return;
        }

        std::string line;
        std::getline(file, line); 

        int total_lines = 0;
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
                std::cerr << "Ligne " << total_lines << " ignorée (mauvais format)" << std::endl;
                continue;
            }
            y.push_back(row.back());
            row.pop_back(); 
            X.push_back(row);  
        }
        std::cout << "Chargé " << X.size() 
                  << " échantillons " 
                  << " de " << filename << std::endl;
    }
};