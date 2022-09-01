cmake_minimum_required(VERSION 3.14)

enable_testing()
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIR})

add_executable(test_all test_all.cpp)

target_link_libraries(test_all ${GTEST_BOTH_LIBRARIES} pthread)

add_test(AllTest, test_all)
