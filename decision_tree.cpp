#include <iostream>
#include <vector>
#include <cmath>

/**
 * @brief Calcule la moyenne d'un vecteur.
 * 
 * @param values Vecteur contenant des nombres.
 * @return La moyenne des valeurs.
 */
double mean(const std::vector<double>& values) {
    double sum = 0.0;
    for(double v : values) {
        sum += v;
    }
    return sum / values.size();
}

/**
 * @brief Calcule l'erreur quadratique moyenne (MSE).
 * 
 * Le MSE mesure à quel point les valeurs s'écartent de la moyenne du nœud.
 * 
 * @param values Vecteur des valeurs cibles (performances).
 * @return Le MSE du vecteur.
 */
double mse(const std::vector<double>& values) {
    double m = mean(values);
    double sum = 0.0;

    for(double v : values) {
        sum += (v - m) * (v - m);
    }
    return sum;
}

/**
 * @brief Structure d'un nœud d'arbre de décision.
 */
struct Node {
    bool is_leaf = false;
    int samples = 0;
    int feature_index = -1;
    double threshold = 0.0;
    double value = 0.0;

    Node* left = nullptr;
    Node* right = nullptr;
};

#define MAX_DEPTH 10
#define MIN_SAMPLES 3
#define MSE_MAX 1e12

/**
 * @brief Construit récursivement un arbre de décision de régression.
 * 
 * @param X Matrice des features (n échantillons × m features)
 * @param y Vecteur de performances (n valeurs)
 * @param depth Profondeur actuelle dans l'arbre
 * @return Un pointeur vers le nœud créé (racine ou sous-noeud)
 */
Node* build_tree(const std::vector<std::vector<double>>& X,
                 const std::vector<double>& y,
                 int depth = 0) 
{
    Node* node = new Node();
    node->samples = y.size();

    // Stop Rules
    if (depth >= MAX_DEPTH || y.size() <= MIN_SAMPLES) {
        node->is_leaf = true;
        node->value = mean(y);
        return node;
    }

    int n = X.size();
    int m = X[0].size();

    double best_mse = MSE_MAX;
    int best_feature = -1;
    double best_threshold = 0.0;

    // On cherche le meilleur split (feature + seuil)
    for (int feature = 0; feature < m; ++feature) {
        for (int row = 0; row < n; ++row) {

            double threshold = X[row][feature];

            std::vector<double> left_y, right_y;

            // Séparation des données selon le threshold
            for (int i = 0; i < n; ++i) {
                if (X[i][feature] <= threshold)  
                    left_y.push_back(y[i]);
                else
                    right_y.push_back(y[i]);
            }

            double mse_split = 
                left_y.size() * mse(left_y) +
                right_y.size() * mse(right_y);

            mse_split /= n;

            if (mse_split < best_mse) {
                best_mse = mse_split;
                best_feature = feature;
                best_threshold = threshold;
            }
        }
    }

    // Si aucun split n'est bon, on stoppe
    if (best_feature == -1) {
        node->is_leaf = true;
        node->value = mean(y);
        return node;
    }

    node->feature_index = best_feature;
    node->threshold = best_threshold;

    //  Séparation finale pour construire les enfants
    std::vector<std::vector<double>> X_left, X_right;
    std::vector<double> y_left, y_right;

    for (int i = 0; i < n; ++i) {
        if (X[i][best_feature] <= best_threshold) {
            X_left.push_back(X[i]);
            y_left.push_back(y[i]);
        } else {
            X_right.push_back(X[i]);
            y_right.push_back(y[i]);
        }
    }

    node->left = build_tree(X_left, y_left, depth + 1);
    node->right = build_tree(X_right, y_right, depth + 1);

    return node;
}

/**
 * @brief Prédit une valeur à partir d'un échantillon en parcourant l'arbre.
 * 
 * @param node Pointeur vers la racine ou un nœud interne.
 * @param sample Vecteur des features d'un échantillon.
 * @return La valeur prédite (moyenne dans une feuille).
 */
double predict(Node* node, const std::vector<double>& sample) {
    if (node->is_leaf) 
        return node->value;

    if (sample[node->feature_index] <= node->threshold)
        return predict(node->left, sample);
    else
        return predict(node->right, sample);
}

/**
 * @brief Programme principal : construit un arbre avec tes données.
 */
int main() {

    // -------------------------
    // TEST 10 échantillons
    // -------------------------
    std::vector<std::vector<double>> test_features = {
        {18,8,4784,4974,43,213,53,5,27,30},
        {22,3,2427,3109,108,29,2,92,4,7},
        {11,9,3426,2805,94,15,98,70,10,21},
        {22,20,4756,2779,153,230,18,32,24,11},
        {22,30,1328,3868,249,197,19,66,31,13},
        {11,18,4167,1397,56,175,42,42,12,31},
        {16,6,1646,2044,69,46,53,97,31,19},
        {8,5,1115,1111,111,20,35,67,22,24},
        {26,13,2668,1348,159,124,28,35,25,13},
        {19,12,2916,3957,49,118,13,72,16,29}
    };

    std::vector<double> test_performance = {
        0.130534, 0.0321952, 0.0374015, 
        0.0576987, 0.0209786, 0.027608, 
        0.0305594, 0.0263897, 0.0271464, 
        0.0403354
    };

   
    Node* my_tree = build_tree(test_features, test_performance);

    double prediction = predict(my_tree, test_features[0]);

   std::cout << "\nPredictions pour toutes les lignes :\n";

   for (size_t i = 0; i < test_features.size(); ++i) {
        double p = predict(my_tree, test_features[i]);
        std::cout << "Ligne " << i+1 << " -> prediction = " << p
              << " | valeur réelle = " << test_performance[i] << std::endl;
}


    return 0;
}

