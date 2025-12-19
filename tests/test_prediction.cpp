#include "DecisionTreeRegressor.hpp"
#include "DataLoader.hpp"
#include <gtest/gtest.h>
#include <cmath>


TEST(DecisionTreeTest, PredictNotNaN) {
   std::vector<std::vector<double>> X = {
       {0.1, 0.2},
       {0.2, 0.3},
       {0.3, 0.4},
       {0.4, 0.5}
   };
   std::vector<double> y = {1.0, 1.5, 2.0, 2.5};


   DecisionTreeRegressor tree;
   tree.fit(X, y);


   double pred = tree.predict({0.25, 0.35});


   EXPECT_FALSE(std::isnan(pred));
   EXPECT_GT(pred, 0.0);
}


int main(int argc, char **argv) {
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}
