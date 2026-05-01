cmake_minimum_required(VERSION 3.16)
project(ProjetPPNME LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_TESTS "Build unit tests" OFF)

include_directories(src)


#note for myself :
# -flto : Link Time Optimization. Permet d'inliner les fonctions même à travers 
#         différents fichiers .cpp, éliminant l'overhead des appels de fonctions.
# -fno-omit-frame-pointer : Interdit au compilateur de supprimer le registre RBP. 
#                           C'est obligatoire pour que AMDuProf et perf 
#                           puissent reconstruire la pile d'appel (Call Stack) sans erreur.

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -flto -g -fno-omit-frame-pointer -Wall -Wextra")

set(PROJECT_SOURCES
    src/main.cpp
    src/DecisionTreeRegressor.cpp
    src/HistogramTreeRegressor.cpp
    src/BaggingRegressor.cpp
    src/BoostingRegressor.cpp
    src/HistogramBaggingRegressor.cpp
    src/HistogramBoostingRegressor.cpp
)
find_package(OpenMP REQUIRED)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(main ${PROJECT_SOURCES})

if(BUILD_TESTS)
    enable_testing()
    find_package(GTest REQUIRED)
    include_directories(${GTEST_INCLUDE_DIRS})

    function(add_gtest_executable exe_name src_file)
        add_executable(${exe_name}
            ${src_file}
            src/DecisionTreeRegressor.cpp
            src/HistogramTreeRegressor.cpp
        )
        target_link_libraries(${exe_name} ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARY} pthread)
        add_test(NAME ${exe_name} COMMAND ${exe_name})
    endfunction()

    add_gtest_executable(test_dataloader tests/test_dataloader.cpp)
    add_gtest_executable(test_mse tests/test_mse.cpp)
    add_gtest_executable(test_prediction tests/test_prediction.cpp)
    add_gtest_executable(test_small_tree tests/test_small_tree.cpp)
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(main PRIVATE OpenMP::OpenMP_CXX)
endif()