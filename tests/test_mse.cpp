#include "DecisionTreeRegressor.hpp"
#include "DataLoader.hpp"
#include <gtest/gtest.h>


TEST(DecisionTreeTest, FitAndPredict) {
   DecisionTreeRegressor tree;


   std::vector<std::vector<double>> X = {
       {1,2,3,4,5,6,7,8,9,10},
       {1,2,3,4,5,6,7,8,9,11},
       {1,2,3,4,5,6,7,8,9,12}
};
   std::vector<double> y = {1.0, 2.0, 3.0};


    tree.fit(X, y);
   double pred = tree.predict(X[0]);


   EXPECT_GE(pred, 1.0);   
   EXPECT_LE(pred, 3.0);    
}


int main(int argc, char **argv) {
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}


