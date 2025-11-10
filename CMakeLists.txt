cmake_minimum_required(VERSION 3.10)

project(PPN)

set(CMAKE_CXX_STANDARD 17)

add_executable(Main src/data_reader.cpp src/decision_tree.cpp src/main.cpp src/node.cpp)
target_include_directories(Main PRIVATE src)

find_package(GTest REQUIRED)