#include "DecisionTreeRegressor.hpp"
#include "DataLoader.hpp"
#include <gtest/gtest.h>
#include <cmath>


TEST(DecisionTreeTest, SmallTreePrediction) {
   // Jeu de données simple
   std::vector<std::vector<double>> X = {{0.0}, {1.0}, {2.0}, {3.0}};
   std::vector<double> y = {0.0, 1.0, 2.0, 3.0};

//créer un petit arbre de décision
   DecisionTreeRegressor tree;
   // Configurer l'arbre pour qu'il soit petit
   tree.max_depth = 3;
   tree.min_samples_split = 1;
   tree.fit(X, y);// Entraînement de l'arbre


   double pred = tree.predict({2.0});// Prédiction pour la valeur 2.0
   // verifier que la prédiction est proche de 2.0
   EXPECT_NEAR(pred, 2.0, 0.5);  // tolérance 0.5
}


int main(int argc, char **argv) {
   ::testing::InitGoogleTest(&argc, argv);// Initialisation de Google Test
   return RUN_ALL_TESTS();// Exécution de tous les tests
}



