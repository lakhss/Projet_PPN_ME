#include "Node_V1.hpp"
#include <iostream>
#include <vector>
#include <cmath>


// for now, we will use mse formula to determine which feature to pick to split
// each node and for each of these features which threshold is the most optimal 

double mean(const std::vector<double>& x){
    double sum = 0;
    for (double i = 0; i < x.size(); ++i){
        sum = x[i]+sum;
    }
    sum /= x.size();
    return sum;
}

double mse(const std::vector<double>& y){ // the formula for mse is the sum of (obtained_value - desired_value)² 
    // for each value of the node, the desired value is the mean of the values within the node 
    double sum = 0;
    double m = mean(y);
    for (auto j = 0; j < y.size(); ++j){
        sum += pow((y[j]-m),2);
    }
    return sum;
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
# define MIN_SAMPLES 1
# define MSE_MAX 100

Node* tree (const std::vector<std::vector<double>>& Features, const std::vector<double>& Performance, int depth = 0){
    // current tree depth is null since we start at the root node
    Node* node = new Node();
    node->samples = Performance.size();

    // stop rule : if DEPTH_MAX is reached or if the number of samples within the node
    // don't go over the MIN_SAMPLES, we stop splitting

    if(depth >= MAX_DEPTH || node->samples <= MIN_SAMPLES){
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

            double threshold = Features[j][i]; // going trough every value of each feature
            std::vector<double> left, right; // storing the values of each performance either on the left node or the right

            for (auto k = 0; k < n ; ++k){
                if(Features[k][i] <= threshold){
                    left.push_back(Performance[k]); 
                }
                else {right.push_back(Performance[k]);}
            }

            if (left.empty() || right.empty()) continue;  

            double mse_ = (left.size()*mse(left) + right.size()*mse(right)) / n;

            if (mse_ < mse_opt){ // if the mse is smaller, we keep it as optimal, along with the feature used and the threshold used
                mse_opt = mse_;
                feature_opt = i;
                threshold_opt = threshold;
            }
        }
    }

    // if there is no mse that is inferior to MSE_MAX, we will make the node a leaf
    if (feature_opt == -1){
        node->is_leaf = true;
        node->value = mean(Performance);
        return node;
    }


    node->feature_idx = feature_opt;
    node->threshold = threshold_opt;


    // Now we split the node into two children node, left and rigth
    std::vector<std::vector<double>> Features_left, Features_right;
    std::vector<double> Performance_left, Performance_right;

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

// to predict new data
double predict(Node* node, const std::vector<double>& x){
    if(node->is_leaf) {
        return node->value;
    } // once we reach a leaf, we return the mean
    
    if(x[node->feature_idx] <= node->threshold) {
        return predict(node->left, x);
    }
    else {
        return predict(node->right, x);
    }
}

void print_vec(const std::vector<double>& x){
    std::cout << "printing performances";
    std::cout << std::endl;
    for (auto i = 0; i < x.size(); ++i){
        std::cout << x[i] << " ";
    }
    std::cout << std::endl;
}

void print_vec_vec(const std::vector<std::vector<double>>& x){
    std::cout << "printing features";
    std::cout << std::endl;
    for (auto i= 0; i< x[0].size(); ++i){
        for (auto j = 0; j<x.size(); ++j){
            std::cout << x[i][j] << " ";
        }
    std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main(){
    std::vector<std::vector<double>> test_features {{18,8,4784,4974,43,213,53,5,27,30},
                                                    {22,3,2427,3109,108,29,2,92,4,7},
                                                    {11,9,3426,2805,94,15,98,70,10,21},
                                                    {22,20,4756,2779,153,230,18,32,24,11},
                                                    {22,30,1328,3868,249,197,19,66,31,13},
                                                    {11,18,4167,1397,56,175,42,42,12,31},
                                                    {16,6,1646,2044,69,46,53,97,31,19},
                                                    {8,5,1115,1111,111,20,35,67,22,24},
                                                    {26,13,2668,1348,159,124,28,35,25,13},
                                                    {19,12,2916,3957,49,118,13,72,16,29}};
    std::vector<double> test_performance {0.130534, 0.0321952, 0.0374015, 0.0576987, 0.0209786, 0.027608, 0.0305594, 0.0263897, 0.0271464, 0.0403354 };
    // print_vec(test_performance);
    // print_vec_vec(test_features);
    Node* my_tree = tree(test_features,test_performance);

    std::vector<double> test_predict {18,30,4276,4748,34,169,96,44,3,10};
    
    double m = mean(test_predict);
    std::cout << "mean = " << m << std::endl;

    double p = predict(my_tree, test_predict);
    std::cout << "value predicted = " << p << std::endl;




}