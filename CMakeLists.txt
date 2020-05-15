cmake_minimum_required (VERSION 3.0)

if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
    message(FETAL_ERROR "GCC version must be at least 9.0")
endif()

project(Tensor_lib)
set(CMAKE_CXX_STANDARD 17)

add_subdirectory(tests)

add_executable(main main.cpp)
