cmake_minimum_required(VERSION 3.14)
project(ProjetPPNME)
set(CMAKE_CXX_STANDARD 17)
include_directories(src)  

add_compile_options(-O3 -march=native -Wall -Wextra)

add_executable(test_loader 
    src/main.cpp 
    src/DecisionTreeRegressor.cpp 
    src/DecisionTreeRegressor.hpp
    src/CrossValidation.hpp
    src/BaggingRegressor.cpp
    src/BaggingRegressor.hpp
    src/BoostingRegressor.cpp
    src/BoostingRegressor.hpp
    src/PerformanceEvaluator.hpp
    src/Node.hpp
    src/DataLoader.hpp
    src/Metrics.hpp
) 


# ===== Google Test =====
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})


# Fonction utilitaire pour créer des exécutables de tests avec GTest
function(add_gtest_executable exe_name src_file extra_cpp)
   add_executable(${exe_name} ${src_file} ${extra_cpp})
   target_link_libraries(${exe_name} ${GTEST_LIBRARIES} pthread)
endfunction()


# ===== Tests unitaires =====
add_gtest_executable(test_dataloader    tests/test_dataloader.cpp src/DecisionTreeRegressor.cpp src/DataLoader.cpp)
add_gtest_executable(test_mse           tests/test_mse.cpp src/DecisionTreeRegressor.cpp)
add_gtest_executable(test_prediction    tests/test_prediction.cpp src/DecisionTreeRegressor.cpp)
add_gtest_executable(test_small_tree    tests/test_small_tree.cpp src/DecisionTreeRegressor.cpp)
