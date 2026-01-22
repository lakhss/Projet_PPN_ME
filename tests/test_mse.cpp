#include "DecisionTreeRegressor.hpp"
#include "DataLoader.hpp"
#include <gtest/gtest.h>

// Test pour le DecisionTreeRegressor
TEST(DecisionTreeTest, FitAndPredict) {
   DecisionTreeRegressor tree;

// Jeu de données simple
   std::vector<std::vector<double>> X = {
       {1,2,3,4,5,6,7,8,9,10},
       {1,2,3,4,5,6,7,8,9,11},
       {1,2,3,4,5,6,7,8,9,12}
};
   std::vector<double> y = {1.0, 2.0, 3.0};// Valeurs cibles correspondantes


    tree.fit(X, y);// Entraînement de l'arbre
   double pred = tree.predict(X[0]);// Prédiction pour le premier échantillon


   EXPECT_GE(pred, 1.0);//s’assure que pred >= 1.0  
   EXPECT_LE(pred, 3.0);//s’assure que pred <= 3.0  
}


int main(int argc, char **argv) {
   ::testing::InitGoogleTest(&argc, argv);// Initialisation de Google Test
   return RUN_ALL_TESTS();// Exécution de tous les tests
}


