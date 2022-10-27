#pragma once

#include <memory>

#include "../tensor.hpp"
#include <gtest/gtest.h>

TEST(TESTND, Test_size) {
  Tensor<double, 10, 10, 10, 10> tensor;
  int size = tensor.get_size();
  EXPECT_EQ(size, 10000);
}

TEST(TESTND, Test_value) {
  Tensor<double, 10, 10, 10, 10> tensor;
  tensor[0][1][2][3] = 10.0;
  tensor[1][2][3][4] = tensor[0][1][2][3] + 20.0;
  EXPECT_EQ(tensor[0][1][2][3], 10.0);
  EXPECT_EQ(tensor[1][2][3][4], 30.0);
}

TEST(TESTND, Test_sum) {
  Tensor<int, 1, 2, 3, 4> tensor(1);
  EXPECT_EQ(tensor.sum(), 24);
}

TEST(TESTND, Test_copy) {
  Tensor<int, 1, 2, 3, 4> tensor(1);
  Tensor<int, 1, 2, 3, 4> tensor_copy{tensor};
  EXPECT_EQ(tensor_copy.sum(), 24);
}

TEST(TESTND, Test_add_scalar) {
  Tensor<int, 1, 2, 3, 4> tensor(1);
  tensor.add(1);
  EXPECT_EQ(tensor.sum(), 48);
}

TEST(TESTND, Test_add_tensor) {
  Tensor<int, 2, 2> src(1);
  Tensor<int, 2, 2> dst(2);
  src += dst;
  EXPECT_EQ(src.sum(), 12);
}