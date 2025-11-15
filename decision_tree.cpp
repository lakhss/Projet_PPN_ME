<<<<<<< HEAD

#include <bits/stdc++.h>
using namespace std;

struct Row { vector<double> x; double y; };

struct Split { int feature; double threshold; double score; // reduction in mse (higher is better)
    vector<int> left_idx; vector<int> right_idx; };

struct Node {
    bool is_leaf = false;
    double value = 0.0; // mean target if leaf
    int feature = -1;
    double threshold = 0.0;
    Node* left = nullptr;
    Node* right = nullptr;
    ~Node(){ delete left; delete right; }
};

pair<double,double> mean_and_mse(const vector<Row>& data, const vector<int>& idx){
    double sum = 0; int n = idx.size();
    for(int i: idx) sum += data[i].y;
    double mean = n? sum / n : 0.0;
    double mse = 0;
    for(int i: idx){ double d = data[i].y - mean; mse += d*d; }
    return {mean, mse};
}


Split find_best_split(const vector<Row>& data, const vector<int>& idx, int min_samples_split){
    Split best; best.feature = -1; best.threshold = 0; best.score = 0; 
    int n = idx.size();
    if(n < 2 * min_samples_split) return best; // no valid split

    int m = data[0].x.size();
   
    auto [parent_mean, parent_mse] = mean_and_mse(data, idx);

    // For each feature, collect (value, index) and sort
    for(int f=0; f<m; ++f){
        vector<pair<double,int>> vals; vals.reserve(n);
        for(int i: idx) vals.emplace_back(data[i].x[f], i);
        sort(vals.begin(), vals.end(), [](auto &a, auto &b){ return a.first < b.first; });

      
        vector<double> pref_y(n+1, 0.0);
        for(int i=0;i<n;++i) pref_y[i+1] = pref_y[i] + data[vals[i].second].y;

      
        for(int s = min_samples_split; s <= n - min_samples_split; ++s){
            double vleft = vals[s-1].first;
            double vright = vals[s].first;
            if(vleft == vright) continue; 
            double thr = (vleft + vright)/2.0;

            int nl = s; int nr = n - s;
            double sumL = pref_y[s]; double sumR = pref_y[n] - sumL;
            double meanL = sumL / nl; double meanR = sumR / nr;
            double mseL = 0, mseR = 0;
            for(int i=0;i<s;++i){ double d = data[vals[i].second].y - meanL; mseL += d*d; }
            for(int i=s;i<n;++i){ double d = data[vals[i].second].y - meanR; mseR += d*d; }

            double weighted_mse = mseL + mseR;
            double reduction = parent_mse - weighted_mse;
            if(reduction > best.score){
                best.score = reduction;
                best.feature = f;
                best.threshold = thr;
               
                best.left_idx.clear(); best.right_idx.clear();
                best.left_idx.reserve(nl); best.right_idx.reserve(nr);
                for(int i=0;i<n;++i){
                    if(data[vals[i].second].x[f] <= thr) best.left_idx.push_back(vals[i].second);
                    else best.right_idx.push_back(vals[i].second);
                }
            }
        }
    }
    return best;
}

Node* build_tree(const vector<Row>& data, const vector<int>& idx, int max_depth, int min_samples_split, int depth=0){
    Node* node = new Node();
    auto [mean, mse] = mean_and_mse(data, idx);
    node->value = mean;
    if(depth >= max_depth || (int)idx.size() < 2*min_samples_split){
        node->is_leaf = true; return node;
    }
    Split s = find_best_split(data, idx, min_samples_split);
    if(s.feature == -1 || s.left_idx.empty() || s.right_idx.empty()){
        node->is_leaf = true; return node;
    }
    node->feature = s.feature;
    node->threshold = s.threshold;
    node->left = build_tree(data, s.left_idx, max_depth, min_samples_split, depth+1);
    node->right = build_tree(data, s.right_idx, max_depth, min_samples_split, depth+1);
    return node;
}

double predict_one(const Node* node, const vector<double>& x){
    const Node* cur = node;
    while(cur && !cur->is_leaf){
        if(x[cur->feature] <= cur->threshold) cur = cur->left;
        else cur = cur->right;
    }
    return cur? cur->value : 0.0;
}

vector<Row> load_csv(const string& path){
    ifstream in(path);
    if(!in) throw runtime_error("Could not open file: " + path);
    string line;
    vector<Row> data;
    
    if(!getline(in, line)) return data;
    auto split_line = [&](const string &ln){
        vector<string> tokens; string cur; bool inquote=false;
        for(char c: ln){
            if(c=='\"') inquote = !inquote;
            else if(c==',' && !inquote){ tokens.push_back(cur); cur.clear(); }
            else cur.push_back(c);
        }
        tokens.push_back(cur);
        return tokens;
    };
    vector<string> first_tokens = split_line(line);
    bool header = false;
   
    auto is_number = [&](const string &s){ if(s.empty()) return false; char* endptr = nullptr; strtod(s.c_str(), &endptr); return *endptr==0; };
    for(const string &t : first_tokens) if(!is_number(t)) { header = true; break; }

    if(!header){
   
        Row r; r.x.clear();
        for(size_t i=0;i+1<first_tokens.size();++i) r.x.push_back(stod(first_tokens[i]));
        r.y = stod(first_tokens.back());
        data.push_back(move(r));
    }
    
    while(getline(in, line)){
        if(line.empty()) continue;
        vector<string> toks = split_line(line);
        if(toks.size() < 2) continue;
        Row r; r.x.clear();
        try{
            for(size_t i=0;i+1<toks.size();++i) r.x.push_back(stod(toks[i]));
            r.y = stod(toks.back());
        } catch(...){
            throw runtime_error("Non-numeric value encountered in CSV (line: '" + line + "')");
        }
        data.push_back(move(r));
    }
    return data;
}

// RMSE
double rmse(const vector<double>& pred, const vector<double>& truth){
    if(pred.size()!=truth.size()) throw runtime_error("Size mismatch in rmse");
    double s=0; for(size_t i=0;i<pred.size();++i){ double d = pred[i]-truth[i]; s += d*d; }
    return sqrt(s/pred.size());
}

int main(int argc, char** argv){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    if(argc < 2){ cerr << "Usage: " << argv[0] << " data.csv [max_depth=10] [min_samples_split=5]\n"; return 1; }
    string path = argv[1];
    int max_depth = 10; int min_samples_split = 5;
    if(argc >= 3) max_depth = stoi(argv[2]);
    if(argc >= 4) min_samples_split = stoi(argv[3]);

    vector<Row> data;
    try{ data = load_csv(path); }
    catch(const exception &e){ cerr << "Error loading CSV: " << e.what() << '\n'; return 2; }
    if(data.empty()){ cerr << "No data found in CSV\n"; return 3; }
    int n = data.size();
    cerr << "Loaded " << n << " rows, " << data[0].x.size() << " features.\n";

   
    vector<int> idx(n); iota(idx.begin(), idx.end(), 0);
    mt19937_64 rng(123456);
    shuffle(idx.begin(), idx.end(), rng);
    int ntrain = max(1, int(0.8 * n));
    vector<int> train_idx(idx.begin(), idx.begin()+ntrain);
    vector<int> test_idx(idx.begin()+ntrain, idx.end());

    Node* root = build_tree(data, train_idx, max_depth, min_samples_split);

    // predictions
    vector<double> preds_train; preds_train.reserve(train_idx.size());
    vector<double> truth_train; truth_train.reserve(train_idx.size());
    for(int i: train_idx){ preds_train.push_back(predict_one(root, data[i].x)); truth_train.push_back(data[i].y); }
    vector<double> preds_test; preds_test.reserve(test_idx.size());
    vector<double> truth_test; truth_test.reserve(test_idx.size());
    for(int i: test_idx){ preds_test.push_back(predict_one(root, data[i].x)); truth_test.push_back(data[i].y); }

    cerr << fixed << setprecision(6);
    cerr << "Train RMSE: " << rmse(preds_train, truth_train) << " (n="<<preds_train.size()<<")\n";
    cerr << "Test  RMSE: " << rmse(preds_test, truth_test) << " (n="<<preds_test.size()<<")\n";

  
    cerr << "\nFirst 5 test predictions (pred -> true):\n";
    for(int i=0;i<min<int>(5, test_idx.size()); ++i){
        int id = test_idx[i];
        double p = predict_one(root, data[id].x);
        cerr << p << " -> " << data[id].y << "\n";
    }

   
    delete root;
    return 0;
}
=======
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>


/**
 * @brief Computes the arithmetic mean of a vector.
 * @param values Vector of numbers.
 * @return Mean of the values.
 * @note Used to predict the value at a leaf and to calculate the variance (MSE).
 */
double mean(const std::vector<double>& values) {
    double sum = 0.0;
    for(double v : values) {
        sum += v;
    }
    return sum / values.size();
}

/**
 * @brief Computes the variance of a vector (Mean Squared Error, MSE).
 * @param values Vector of real values.
 * @return Variance = mean of squared differences from the mean. Returns 0 if empty.
 * @note Used to evaluate the quality of a split in the decision tree.
 */
double mse(const std::vector<double>& values) {
    if (values.empty()) return 0.0;
    double m = mean(values);
    double sum = 0.0;
    for (double v : values) {
        double d = v - m;
        sum += d * d;
    }
    return sum / static_cast<double>(values.size());

}
/**
 * @brief Decision tree node.
 * 
 * is_leaf: true if leaf.
 * samples: number of samples in the node.
 * feature_index: feature used for split (-1 if leaf).
 * threshold: split value.
 * value: predicted value if leaf.
 * left/right: child nodes.
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
 * @brief Recursively builds a regression decision tree.
 * 
 * @param X Feature matrix (n samples × m features)
 * @param y Vector of target values (n values)
 * @param depth Current depth in the tree
 * @return Pointer to the created node (root or subtree)
 */
Node* build_tree(const std::vector<std::vector<double>>& X,
                 const std::vector<double>& y,
                 int depth = 0)
{
    Node* node = new Node();
    node->samples = y.size();

    double current_mse = mse(y);

    // Stop conditions: max depth, few samples, or very small variance
    if (depth >= MAX_DEPTH || y.size() <= MIN_SAMPLES || current_mse < 1e-6) {
        node->is_leaf = true;
        node->value = mean(y);
        return node;
    }

    int n = X.size();        // number of samples
    int m = X[0].size();    // number of features

    double best_mse = MSE_MAX;
    int best_feature = -1;
    double best_threshold = 0.0;

    for (int feature = 0; feature < m; ++feature) {

         // Collect and sort all values of this feature
        std::vector<double> sorted_values;
        sorted_values.reserve(n);

        for (auto &row : X)
            sorted_values.push_back(row[feature]);

        std::sort(sorted_values.begin(), sorted_values.end());

        // Test thresholds between consecutive values
        for (int i = 0; i < n - 1; ++i) {

            double threshold = (sorted_values[i] + sorted_values[i+1]) / 2.0;

            std::vector<double> left_y, right_y;
             

            // Split target values according to threshold
            for (int k = 0; k < n; ++k) {
                if (X[k][feature] <= threshold)
                    left_y.push_back(y[k]);
                else
                    right_y.push_back(y[k]);
            }

            if (left_y.empty() || right_y.empty())
                continue;
            
            // Compute weighted MSE for this split
            double mse_split =
                (left_y.size() * mse(left_y) +
                 right_y.size() * mse(right_y)) / n;

            if (mse_split < best_mse) {
                best_mse = mse_split;
                best_feature = feature;
                best_threshold = threshold;
            }
        }
    }

    if (best_feature == -1) {
        node->is_leaf = true;
        node->value = mean(y);
        return node;
    }

    node->feature_index = best_feature;
    node->threshold = best_threshold;

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
 * @brief Predicts a value for a sample by traversing the tree.
 * 
 * @param node Pointer to the root or current node.
 * @param sample Vector of features for a single sample.
 * @return Predicted value (mean stored in a leaf).
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
 * @brief Main program: builds a decision tree with test data.
 */
int main() {


     // TEST 10 samples 
     
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

    // === Second test ===
    std::vector<std::vector<double>> test_features2 = {
        {17,6,2396,4667,182,25,16,40,31,18},
        {5,27,3047,4121,145,194,3,55,1,5},
        {8,0,1616,3255,106,68,23,78,26,27},
        {2,17,1289,2160,115,193,31,47,28,10},
        {2,4,4171,3115,137,11,68,47,13,27},
        {28,7,1710,3916,80,208,15,3,26,24},
        {24,26,1234,1252,70,35,73,92,20,1},
        {9,14,3150,1625,249,198,50,5,7,28},
        {11,11,3154,4932,189,169,23,41,20,9},
        {10,8,3273,2260,81,253,58,85,26,27}
        // tu peux ajouter toutes les autres lignes ici de la même manière
    };
    std::vector<double> test_performance2 = {
        0.198022,0.0549579,0.0212354,0.0229885,0.145877,
        0.0329989,0.0197994,0.0355665,0.0535637,0.0434353
        // idem, ajoute toutes les valeurs de performance correspondantes
    };

    Node* tree2 = build_tree(test_features2, test_performance2);

    std::cout << "\nPredictions for second test set:\n";
    for (size_t i = 0; i < test_features2.size(); ++i) {
        double p = predict(tree2, test_features2[i]);
        std::cout << "Sample " << i+1
                  << " -> predicted = " << p
                  << " | actual = " << test_performance2[i] << "\n";
    }


    return 0;
}

>>>>>>> origin/main
