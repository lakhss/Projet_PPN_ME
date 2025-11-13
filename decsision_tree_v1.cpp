#include "nodetest.hpp"

// need to add a struct to choose which method to use to pick which feature is the most optimal to split the node?

// We can choose here which method will be used to measure "purity", to determine which value of the feature to take
struct Method {
    double mse(const vector<double>& y){} // mean squared error
    double var(const vector<double>& y){} // variance 
};

// We can choose here which rule to use to stop partitionning, to avoid over-fitting
struct Stop_rule {
    int depth(int depth){} // stops when the tree reaches a given depth
    void greedy(){} // doesn't stop until all samples belong to the same class
    void no_samples(){} // no samples left
    void min(){} // has to be a minimum of values of perf in the leaves

};

Node* tree (const vector<vector<double>>& Features, const vector<double>& Performance, Method *method, Stop_rule *stop, Node *node){
    Node* node = new Node();

    int n = Features.size();
    int m = Features[0].size();
    // add the method
    // add the stop rule
    

    for (int feature = 0; feature < m; ++feature){
        for (int i = 0; i<n; ++i){
            node->threshold = Feature[i][feature]; // going trough every column of features

            // loop to calculate which feature gives the best result for the chosen method

            
            }
        }
    }
    
}