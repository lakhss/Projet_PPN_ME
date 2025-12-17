#include "DecisionTreeRegressor.hpp"
#include "DataLoader.hpp"
#include <gtest/gtest.h>
#include <cmath>


TEST(DecisionTreeTest, SmallTreePrediction) {
   std::vector<std::vector<double>> X = {{0.0}, {1.0}, {2.0}, {3.0}};
   std::vector<double> y = {0.0, 1.0, 2.0, 3.0};


   DecisionTreeRegressor tree;
   tree.max_depth = 3;
   tree.min_samples_split = 1;
   tree.fit(X, y);


   double pred = tree.predict({2.0});
   EXPECT_NEAR(pred, 2.0, 0.5);  // tolérance 0.5
}


int main(int argc, char **argv) {
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}



