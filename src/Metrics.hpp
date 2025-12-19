#ifndef METRICS_HPP
#define METRICS_HPP


#include <vector>
#include <cmath>
#include <iostream>
#include <numeric>

class Metrics {
    public:
    // Mean Squared Error (Erreur Quadratique Moyenne)
    static double mean_squared_error(const std::vector<double>& y_true,
                                         const std::vector<double>& y_pred) {
        if ( y_true.size() != y_pred.size() || y_true.empty()) return 0.0;

        double sum_sq_err = 0.0;
            for (size_t i = 0; i < y_true.size(); ++i) {
                double diff = y_true[i] - y_pred[i];
                sum_sq_err += diff * diff;
            }
        return sum_sq_err / y_true.size();
    }

    // Root Mean Squared Error (Racine Carrée de l'Erreur Quadratique Moyenne)
    static double root_mean_squared_error(const std::vector<double>& y_true, 
                                          const std::vector<double>& y_pred) {
        return std::sqrt(mean_squared_error(y_true, y_pred));
    }

    // Mean Absolute Error (Moyenne des Erreurs Absolues)
    static double mean_absolute_error(const std::vector<double>& y_true, 
                                      const std::vector<double>& y_pred) {
        if (y_true.size() != y_pred.size() || y_true.empty()) return 0.0;

        double sum_abs_err = 0.0;
        for (size_t i = 0; i < y_true.size(); ++i) {
            sum_abs_err += std::abs(y_true[i] - y_pred[i]);
        }
        return sum_abs_err / y_true.size();
    }

    static double mean_absolute_percentage_error(const std::vector<double>& y_true, 
                                                 const std::vector<double>& y_pred) {
        if (y_true.size() != y_pred.size() || y_true.empty()) return 0.0;

        double sum_ape = 0.0;
        double epsilon = 1e-9; // Sécurité pour éviter la division par 0

        for (size_t i = 0; i < y_true.size(); ++i) {
            double denominator = std::abs(y_true[i]);
            if (denominator < epsilon) denominator = epsilon; // On évite le crash
            
            sum_ape += std::abs((y_true[i] - y_pred[i]) / denominator);
        }
        
        // On retourne un pourcentage 
        return (sum_ape / y_true.size()) * 100.0; 
    }
};

#endif