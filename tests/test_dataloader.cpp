#include "DecisionTreeRegressor.hpp"
#include "DataLoader.hpp"
#include <gtest/gtest.h>


TEST(DataLoaderTest, LoadCSV) {
   std::vector<std::vector<double>> X;
   std::vector<double> y;


   DataLoader::load_csv("../datasets/15k_ga_adaptive.csv", X, y);


   EXPECT_FALSE(X.empty());              
   EXPECT_EQ(X.size(), y.size());        


   for (const auto& row : X) {
       EXPECT_EQ(row.size(), 10);        
   }
}


int main(int argc, char **argv) {
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}


