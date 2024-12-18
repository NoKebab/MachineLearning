set(target FeedForwardNeuralNet)

set(sources "NeuralNet.cpp" "NeuralNetBuilder.cpp" "NeuralNetActivation.cpp")
set(headers "NeuralNet.hpp" "NeuralNetBuilder.hpp" "NeuralNetActivation.hpp")

add_library(${target} STATIC ${sources} ${headers})

target_link_libraries(${target} LinearAlgebraLib)

target_include_directories(${target} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})