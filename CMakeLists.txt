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
    src/Node.hpp
    src/DataLoader.hpp
    src/Metrics.hpp
) 