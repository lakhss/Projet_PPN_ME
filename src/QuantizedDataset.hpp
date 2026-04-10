#ifndef QUANTIZED_DATASET_HPP
#define QUANTIZED_DATASET_HPP

#include <vector>
#include <cstdint>
#include <string>
#include <stdexcept>
#include <cmath>
#include <algorithm>


#include "Matrix.hpp"



struct FeatureMeta {
    float min_val;
    float max_val;
    float bin_width;
};

class QuantizedDataset {
private:
    size_t num_samples_;
    size_t num_features_;

    // ARCHITECTURE SoA (Structure of Arrays)
    // L'index extérieur est la FEATURE (colonne).
    // L'index intérieur est le SAMPLE (ligne).
    // Utilisation de uint8_t pour maximiser la densité en cache L1/L2 (64 valeurs par cache line)
    std::vector<std::vector<uint8_t>> binned_features_;
    
    std::vector<float> targets_;

    // Métadonnées pour chaque feature afin de pouvoir retraduire un "bin" en "float"
    // si nécessaire lors de l'inférence ou de l'analyse du split.
    std::vector<FeatureMeta> meta_data_;

public:
    QuantizedDataset() : num_samples_(0), num_features_(0) {}

    inline size_t get_num_samples() const { return num_samples_; }
    inline size_t get_num_features() const { return num_features_; }
    
    // Retourne une référence constante sur le tableau contigu d'une feature complète
    // prochaine etape on va etudier ce vecteur et essayer de vectoriser Avec les instructions SIMD 
    inline const std::vector<uint8_t>& get_feature_column(size_t feature_idx) const {
        return binned_features_[feature_idx];
    }

    inline const std::vector<float>& get_targets() const {
        return targets_;
    }

    inline const FeatureMeta& get_meta(size_t feature_idx) const {
        return meta_data_[feature_idx];
    }

    void ingest_and_quantize(const std::vector<std::vector<float>>& raw_csv_data, const std::vector<float>& raw_targets) {
        if (raw_csv_data.empty()) return;
        
        num_samples_ = raw_csv_data.size();
        num_features_ = raw_csv_data[0].size();
        
        targets_ = raw_targets;
        binned_features_.resize(num_features_, std::vector<uint8_t>(num_samples_));
        meta_data_.resize(num_features_);

        // La boucle d'ingestion : on itère colonne par colonne
        for (size_t f = 0; f < num_features_; ++f) {
            float min_val = raw_csv_data[0][f];
            float max_val = raw_csv_data[0][f];

            // Trouver le Min et le Max de la featureee
            for (size_t i = 1; i < num_samples_; ++i) {
                float val = raw_csv_data[i][f];
                if (val < min_val) min_val = val;
                if (val > max_val) max_val = val;
            }

            meta_data_[f].min_val = min_val;
            meta_data_[f].max_val = max_val;
            
            float range = max_val - min_val;
            if (range == 0.0f) {
                range = 1e-7f; 
            }
            meta_data_[f].bin_width = range / 255.0f;

            // Discrétisation (Binning) et stockage SoA
            for (size_t i = 0; i < num_samples_; ++i) {
                float val = raw_csv_data[i][f];
                // Application de la formule mathématique
                float normalized = (val - min_val) / range;
                // Saturation de sécurité entre 0 et 255
                int bin = static_cast<int>(std::floor(normalized * 255.0f));
                if (bin < 0) bin = 0;
                if (bin > 255) bin = 255;
                
                binned_features_[f][i] = static_cast<uint8_t>(bin);
            }
        }
    }
    
    void ingest_from_matrix(const Matrix& raw_X, const std::vector<double>& raw_targets) {
        if (raw_X.empty()) return;
        
        num_samples_ = raw_X.rows();
        num_features_ = raw_X.cols();
        
        // C'est ici que l'on bascule de double (8 octets) à float (4 octets) pour la target
        targets_.assign(raw_targets.begin(), raw_targets.end());
        binned_features_.resize(num_features_, std::vector<uint8_t>(num_samples_));
        meta_data_.resize(num_features_);

        // Laboucle de Transposition Row-Major (Matrix) -> Column-Major (SoA)
        for (size_t f = 0; f < num_features_; ++f) {
            double min_val = raw_X(0, f);
            double max_val = raw_X(0, f);

            // recherche Min/Max (Penalite Row-Major subie une seule fois au chargement)
            for (size_t i = 1; i < num_samples_; ++i) {
                double val = raw_X(i, f);
                if (val < min_val) min_val = val;
                if (val > max_val) max_val = val;
            }

            meta_data_[f].min_val = static_cast<float>(min_val);
            meta_data_[f].max_val = static_cast<float>(max_val);
            
            double range = max_val - min_val;
            if (range == 0.0) range = 1e-7;
            meta_data_[f].bin_width = static_cast<float>(range / 255.0);

            // Discrétisation et écriture SoA contiguë
            for (size_t i = 0; i < num_samples_; ++i) {
                double val = raw_X(i, f);
                double normalized = (val - min_val) / range;
                int bin = static_cast<int>(std::floor(normalized * 255.0));
                if (bin < 0) bin = 0;
                if (bin > 255) bin = 255;
                
                binned_features_[f][i] = static_cast<uint8_t>(bin);
            }
        }
    }
};

#endif 