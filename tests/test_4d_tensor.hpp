#pragma once

#include <gtest/gtest.h>
#include "../tensor.hpp"

TEST(TEST4D, Test_size) {
    Tensor<double, 10, 10, 10, 10> tensor;
    int size = tensor.get_size();
    EXPECT_EQ(size, 10000);
    size = Tensor<double, 10, 10, 10, 10>::get_size();
    EXPECT_EQ(size, 10000);
}

TEST(TEST4D, Test_value) {
    Tensor<double, 10, 10, 10, 10> tensor;
    tensor[0][1][2][3] = 10.0;
    tensor[1][2][3][4] = tensor[0][1][2][3] + 20.0; 
    EXPECT_EQ(tensor[0][1][2][3], 10.0);
    EXPECT_EQ(tensor[1][2][3][4], 30.0);
}
