cmake_minimum_required(VERSION 3.20)

set(target LossMeasuresLib)

set(sources "Loss.cpp")
set(headers "Loss.hpp")

add_library(${target} STATIC ${sources} ${headers})

target_include_directories(${target} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(${target} LinearAlgebraLib)
