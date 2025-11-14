#include "Node_V1.hpp"
#include <vector>

// for now, we will use mse formula to determine which feature to pick to split
// each node and for each of these features which threshold is the most optimal 

double mean(const vector<double>& x){
    double sum = 0;
    for (auto i = x.begin(); i < x.end(); ++i){
        sum += i;
    }
    sum /= x.size();
    return sum;
}

double mse(const vector<double>& y){ // the formula for mse is the sum of (obtained_value - desired_value)² 
    // for each value of the node, the desired value is the mean of the values within the node 
    double sum = 0;
    double m = mean(y);
    for (auto j = y.begin(); j < y.end(), ++j){
        sum += pow((j-m),2);
    }
}

/* 
// We can choose here which rule to use to stop partitionning, to avoid over-fitting
struct Stop_rule {
    int depth(int depth){} // stops when the tree reaches a given depth
    void greedy(){} // doesn't stop until all samples belong to the same class
    void no_samples(){} // no samples left
    void min(){} // has to be a minimum of values of perf in the leaves

};
*/ 

# define MAX_DEPTH 25
# define MIN_SAMPLES 5
# define MSE_MAX 100

Node* tree (const vector<vector<double>>& Features, const vector<double>& Performance, int depth = 0){
    // current tree depth is null since we start at the root node
    Node* node = new Node();
    node->samples = Performance.size();

    // stop rule : if DEPTH_MAX is reached or if the number of samples within the node
    // don't go over the MIN_SAMPLES, we stop splitting

    if(depth >= MAX_DEPTH || node.samples <= MIN_SAMPLES){
        node->is_leaf = true ; // we turn the node into a leaf, to stop splitting it
        node->value = mean(Performance); // the value of the leaf is the mean of all the samples within the leaf
        return node;
    }

    int n = Features.size(); // number of rows of the vector features
    int m = Features[0].size(); // number of columns aka the number of features
    double mse_opt = MSE_MAX; // we will store the most optimal mse value here
    int feature_opt = -1; // we will store the most optimal feature here
    double threshold_opt = 0; // we will store the best threshold here

    for (auto i = 0; i < m; ++i){
        for (auto j = 0; j<n; ++j){

            double threshold = Feature[j][i]; // going trough every value of each feature
            vector<double> left, right; // storing the values of each performance either on the left node or the right

            for (auto k = 0; k < n ; ++k){
                if(Features[k][i] <= threshold){
                    left.push_back(Performance[j]); 
                }
                else {right.push_back(Performance[j]);}
            }

            double mse_ = (left.size()*mse(left) + right.size()*mse(right)) / n;

            if (mse_ < mse_opt){ // if the mse is smaller, we keep it as optimal, along with the feature used and the threshold used
                mse_opt = mse_;
                feature_opt = feature;
                threshold_opt = threshold;
            }
        }
    }

    // if there is no mse that is inferior to MSE_MAX, we will make the node a leaf
    if (best_feature == -1){
        node->is_leaf = true;
        node->value = mean(Performance);
        return node;
    }


    node->feature_idx = feature_opt;
    node->threshold = threshold_opt;


    // Now we split the node into two children node, left and rigth
    vector<vector<double>> Features_left, Features_right;
    vector<double> Performance_left, Performance_right;

    for (auto i = 0; i < n; ++i){
        if(Features[i][feature_opt] <= threshold_opt ){
            Features_left.push_back(Features[i]);
            Performance_left.push_back(Performance[i]);
        }
        else {
            Features_right.push_back(Features[i]);
            Performance_right.push_back(Performance[i]);
        }
    }

    node->left = tree(Features_left, Performance_left, depth+1);
    node->right = tree(Features_right, Performance_right, depth+1);
    return node;

}

// predicting new data
    
double predict(Node* node, const vector<double>& x){
    if(node->is_leaf {
        return node->value;
    } // once we reach a leaf, we return the mean
    
    if(x[node->feature_idx] <= node->threshold) {
        return predict(node->left, x);
    }
    else {
        return predict(node->right, x);
    }
}