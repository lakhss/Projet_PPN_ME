struct Node {
    bool is_leaf = true; // is the node a leaf or not
    double value = 0.0;  // the mean of the performances in the node      
    int feature_idx = -1; // the index of the feature we are using to split this node, goes from value 0 to 10      
    double threshold = 0.0; // the best value to give the feature to split the node    
    int samples; // number of samples within the node
    Node* left = nullptr; 
    Node* right = nullptr;

    ~Node() { delete left; delete right; }
};