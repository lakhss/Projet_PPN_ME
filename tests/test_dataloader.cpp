#include "DecisionTreeRegressor.hpp"
#include "DataLoader.hpp"
#include <gtest/gtest.h>

// Test pour le DataLoader
TEST(DataLoaderTest, LoadCSV) {
   std::vector<std::vector<double>> X;
   std::vector<double> y;

// Chargement des données depuis le fichier CSV
   DataLoader::load_csv("../datasets/15k_ga_adaptive.csv", X, y);


   EXPECT_FALSE(X.empty());// Vérifie que X n'est pas vide 
   EXPECT_EQ(X.size(), y.size());// Vérifie que X et y ont le même nombre d'échantillons        

// Vérifie que chaque échantillon a 10 caractéristiques
   for (const auto& row : X) {
       EXPECT_EQ(row.size(), 10);        
   }
}


int main(int argc, char **argv) {
   ::testing::InitGoogleTest(&argc, argv);// Initialisation de Google Test
   return RUN_ALL_TESTS();// Exécution de tous les tests
}


