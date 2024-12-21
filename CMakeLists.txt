cmake_minimum_required(VERSION 3.20)

project(NeuralNet)

set(target ${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(DataProcessing)
add_subdirectory(LinearAlgebra)
add_subdirectory(FeedForwardNeuralNet)
add_subdirectory(LossMeasures)
add_subdirectory(App)