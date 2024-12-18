set(target LinearAlgebraLib)

set(sources "Matrix2D.cpp")
set(headers "Matrix2D.hpp")

add_library(${target} STATIC ${sources} ${headers})

target_include_directories(${target} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})