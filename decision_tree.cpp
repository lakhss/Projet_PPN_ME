
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
