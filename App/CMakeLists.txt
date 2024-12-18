set(target FeedForwardNerualNetApp)

set(sources "NeuralNetApp.cpp")
set(headers )

add_executable(${target} ${sources} ${headers})

target_link_libraries(${target} PUBLIC FeedForwardNeuralNet DataProcessingLib LossMeasuresLib)

target_include_directories(${target} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
