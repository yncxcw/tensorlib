#pragma once

#include "../tensor.hpp"
#include <gtest/gtest.h>

TEST(TEST1D, Test_size) {
  Tensor<double, 10> tensor;
  int size = tensor.get_size();
  EXPECT_EQ(size, 10);
}

TEST(TEST1D, Test_value) {
  Tensor<double, 10> tensor;
  tensor[0] = 10.0;
  tensor[1] = tensor[0] + 20.0;
  EXPECT_EQ(tensor[0], 10.0);
  EXPECT_EQ(tensor[1], 30.0);
}

TEST(TEST1D, Test_sum) {
  Tensor<double, 10> tensor(0.1);
  EXPECT_DOUBLE_EQ(tensor.sum(), 1.0);
}
