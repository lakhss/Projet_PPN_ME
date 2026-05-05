#pragma once

#include "CrossValidation.hpp"
#include "Metrics.hpp"
#include "Matrix.hpp"


#include "QuantizedDataset.hpp"

#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <functional>
#include <cmath>
#include <string>
#include <fstream>




struct EvalResult {
    double mse;
    double rmse;
    double mae;
    double mape;
    double time_sec;
};

class PerformanceEvaluator {
public:
    static void log_to_csv(const std::string& dataset_name,
                           const std::string& model_name,
                           const EvalResult& res) {
        std::ofstream outfile("results.csv", std::ios::app);
        if (outfile.is_open()) {
            outfile << dataset_name << ","
                    << model_name << ","
                    << res.rmse << ","
                    << res.mae << ","
                    << res.mape << ","
                    << res.time_sec << "\n";
            outfile.close();
        }
    }

    static void get_subset(const Matrix& X_full,
                           const std::vector<double>& y_full,
                           const std::vector<size_t>& indices,
                           Matrix& X_out,
                           std::vector<double>& y_out) {
        y_out.clear();
        y_out.reserve(indices.size());

        if (indices.empty()) {
            X_out.resize(0, X_full.cols());
            return;
        }

        X_out.resize(indices.size(), X_full.cols());

        for (size_t r = 0; r < indices.size(); ++r) {
            size_t i = indices[r];
            for (size_t c = 0; c < X_full.cols(); ++c) {
                X_out(r, c) = X_full(i, c);
            }
            y_out.push_back(y_full[i]);
        }
    }

    template <typename ModelCreator>
    static EvalResult run_cv(ModelCreator model_creator,
                             const Matrix& X,
                             const std::vector<double>& y,
                             int k_folds) {
        if (X.empty() || y.empty() || X.rows() != y.size() || k_folds <= 0) {
            return {0.0, 0.0, 0.0, 0.0, 0.0};
        }

        auto folds = CrossValidation::k_fold_split(X.rows(), k_folds, 42);

        double t_rmse = 0.0;
        double t_mae = 0.0;
        double t_mape = 0.0;
        double t_time = 0.0;

        for (int k = 0; k < k_folds; ++k) {
            Matrix X_train, X_test;
            std::vector<double> y_train, y_test;

            get_subset(X, y, folds[k].train_idx, X_train, y_train);
            get_subset(X, y, folds[k].test_idx, X_test, y_test);

            auto model = model_creator();

            auto start = std::chrono::high_resolution_clock::now();
            model.fit(X_train, y_train);
            auto end = std::chrono::high_resolution_clock::now();

            std::vector<double> preds;
            preds.reserve(X_test.rows());

            for (size_t i = 0; i < X_test.rows(); ++i) {
                preds.push_back(model.predict(X_test.row(i)));
            }

            double mse = Metrics::mean_squared_error(y_test, preds);
            t_rmse += std::sqrt(mse);
            t_mae += Metrics::mean_absolute_error(y_test, preds);
            t_mape += Metrics::mean_absolute_percentage_error(y_test, preds);
            t_time += std::chrono::duration<double>(end - start).count();
        }

        return {0.0,
                t_rmse / k_folds,
                t_mae / k_folds,
                t_mape / k_folds,
                t_time / k_folds};
    }

    template <typename ModelCreator>
    static void evaluate(const std::string& dataset_name,
                         const std::string& model_name,
                         ModelCreator model_creator,
                         const Matrix& X,
                         const std::vector<double>& y,
                         int k_folds = 5) {

        EvalResult res = run_cv(model_creator, X, y, k_folds);

        std::cout << std::fixed << std::setprecision(6);
        std::cout << std::left
                  << std::setw(30) << model_name
                  << std::setw(12) << res.rmse
                  << std::setw(12) << res.mape
                  << std::setw(12) << res.time_sec << "s"
                  << std::endl;

        log_to_csv(dataset_name, model_name, res);
    }


    template <typename ModelCreator>
    static EvalResult run_cv_hpc(ModelCreator model_creator,
                                 const Matrix& X_full,
                                 const std::vector<double>& y_full,
                                 int k_folds) {
        if (X_full.empty() || y_full.empty() || X_full.rows() != y_full.size() || k_folds <= 0) {
            return {0.0, 0.0, 0.0, 0.0, 0.0};
        }

        auto folds = CrossValidation::k_fold_split(X_full.rows(), k_folds, 42);

        double t_rmse = 0.0;
        double t_mae = 0.0;
        double t_mape = 0.0;
        double t_time = 0.0;

        for (int k = 0; k < k_folds; ++k) {
            Matrix X_train, X_test;
            std::vector<double> y_train, y_test;

            // extraction standard (AoS) pour le split Train/Test
            get_subset(X_full, y_full, folds[k].train_idx, X_train, y_train);
            get_subset(X_full, y_full, folds[k].test_idx, X_test, y_test);

            // conversion vers l'Architecture SoA (QuantizedDataset)
            QuantizedDataset q_train;
            q_train.ingest_from_matrix(X_train, y_train);

            auto model = model_creator();

            auto start = std::chrono::high_resolution_clock::now();
            model.fit(q_train); // Appel sur la structure SoA
            auto end = std::chrono::high_resolution_clock::now();

            // Phase d'Inférence (Prédiction)
            std::vector<double> preds;
            preds.reserve(X_test.rows());

            // L'inférence se fait ligne par ligne (Individu par individu)
            for (size_t i = 0; i < X_test.rows(); ++i) {
                preds.push_back(model.predict(X_test.row(i)));
            }

            double mse = Metrics::mean_squared_error(y_test, preds);
            t_rmse += std::sqrt(mse);
            t_mae += Metrics::mean_absolute_error(y_test, preds);
            t_mape += Metrics::mean_absolute_percentage_error(y_test, preds);
            t_time += std::chrono::duration<double>(end - start).count();
        }

        return {0.0,
                t_rmse / k_folds,
                t_mae / k_folds,
                t_mape / k_folds,
                t_time / k_folds};
    }

    template <typename ModelCreator>
    static void evaluate_hpc(const std::string& dataset_name,
                             const std::string& model_name,
                             ModelCreator model_creator,
                             const Matrix& X,
                             const std::vector<double>& y,
                             int k_folds = 5) {
        
        EvalResult res = run_cv_hpc(model_creator, X, y, k_folds);

        std::cout << std::fixed << std::setprecision(6);
        std::cout << std::left
                  << std::setw(30) << model_name
                  << std::setw(12) << res.rmse
                  << std::setw(12) << res.mape
                  << std::setw(12) << res.time_sec << "s"
                  << std::endl;

        log_to_csv(dataset_name, model_name, res);
    }

    static void print_header() {
        std::cout << std::string(60, '-') << std::endl;
        std::cout << std::left
                  << std::setw(30) << "Modele"
                  << std::setw(12) << "RMSE"
                  << std::setw(12) << "MAPE"
                  << std::setw(12) << "Temps"
                  << std::endl;
        std::cout << std::string(60, '-') << std::endl;
    }
};