struct Node {
    bool is_leaf = true;
    double value = 0.0;          
    int feature_idx = -1;       
    double threshold = 0.0;     
    Node* left = nullptr;
    Node* right = nullptr;

    ~Node() { delete left; delete right; }
};