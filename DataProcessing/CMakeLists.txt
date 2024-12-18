set(target DataProcessingLib)

set(sources "Dataset.cpp")
set(headers "Dataset.hpp")

add_library(${target} STATIC ${sources} ${headers})

target_include_directories(${target} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})