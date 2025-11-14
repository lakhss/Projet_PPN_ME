#include <iostream>
#include <vector>
#include <numeric>
using namespace std;

//  STRUCTURE D’UN NŒUD
struct Node {
    bool is_leaf;
    double value;        // valeur prédite si feuille
    int feature;         // colonne utilisée pour le split
    double threshold;    // seuil du split

    Node* left;
    Node* right;

    Node() : is_leaf(false), value(0), feature(-1), threshold(0),
             left(nullptr), right(nullptr) {}
};


double mean(const vector<double>& y){
    double s = 0;
    for(double v : y) s += v;
    return s / y.size();
}


double variance(const vector<double>& y){
    double m = mean(y);
    double s = 0;
    for(double v : y) s += (v - m) * (v - m);
    return s / y.size();
}


//  CONSTRUCTION DE L’ARBRE
Node* build_tree(const vector<vector<double>>& X,
                 const vector<double>& y,
                 int depth)
{
    // règle d'arrêt : peu d'échantillons ou profondeur élevée
    if (y.size() <= 2 || depth >= 3) {
        Node* leaf = new Node();
        leaf->is_leaf = true;
        leaf->value = mean(y);
        return leaf;
    }

    int n = X.size();
    int m = X[0].size();

    int best_feature = -1;
    double best_threshold = 0;
    double best_score = 1e18;

    // On cherche un bon split simple
    for(int f=0; f<m; f++){
        // seuil simple = moyenne du feature
        double thr = 0;
        for(int i=0; i<n; i++) thr += X[i][f];
        thr /= n;

        vector<double> left_y, right_y;

        for(int i=0; i<n; i++){
            if (X[i][f] <= thr) left_y.push_back(y[i]);
            else right_y.push_back(y[i]);
        }

        if (left_y.empty() || right_y.empty()) continue;

        double score = variance(left_y) + variance(right_y);

        if (score < best_score) {
            best_score = score;
            best_feature = f;
            best_threshold = thr;
        }
    }

    // aucun split trouvé : feuille
    if (best_feature == -1) {
        Node* leaf = new Node();
        leaf->is_leaf = true;
        leaf->value = mean(y);
        return leaf;
    }

    // Séparation des données
    vector<vector<double>> X_left, X_right;
    vector<double> y_left, y_right;

    for(int i=0; i<n; i++){
        if (X[i][best_feature] <= best_threshold) {
            X_left.push_back(X[i]);
            y_left.push_back(y[i]);
        } else {
            X_right.push_back(X[i]);
            y_right.push_back(y[i]);
        }
    }

    // Création du nœud
    Node* node = new Node();
    node->feature = best_feature;
    node->threshold = best_threshold;

    node->left = build_tree(X_left, y_left, depth+1);
    node->right = build_tree(X_right, y_right, depth+1);

    return node;
}

//  PRÉDICTION
double predict(Node* node, const vector<double>& sample){
    if (node->is_leaf) return node->value;

    if (sample[node->feature] <= node->threshold)
        return predict(node->left, sample);
    else
        return predict(node->right, sample);
}

// Fonctions d'affichage très simples
void print_vec(const vector<double>& v) {
    for(double x : v) cout << x << " ";
    cout << endl;
}

void print_vec_vec(const vector<vector<double>>& M) {
    for(const auto& row : M) {
        print_vec(row);
    }
}

int main() {

    vector<vector<double>> test_features = {
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

    vector<double> test_performance = {
        0.130534,0.0321952,0.0374015,0.0576987,0.0209786,
        0.027608,0.0305594,0.0263897,0.0271464,0.0403354
    };

    cout << "Features" << endl;
    print_vec_vec(test_features);

    cout << "\nPerformance " << endl;
    print_vec(test_performance);

   
    Node* arbre = build_tree(test_features, test_performance, 0);
    vector<double> nouveau_sample = test_features[0];
    double valeur_predite = predict(arbre, nouveau_sample);

    cout << "\n Prédiction" << endl;
   for(int i=0; i<10; i++){
    double pred = predict(arbre, test_features [i]);
    cout << "Sample " << i+1 << " -> Valeur prédite = " << pred << endl;
    }
    return 0;
}
//g++ arbre_decision.cpp -o arbre
//./arbre