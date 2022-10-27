#include "tensor.hpp"
#include <iostream>

int main() {
  // 1D tensor
  Tensor<double, 20> t1;
  t1[4] = 10.0;
  t1[5] = t1[4] + 1;
  std::cout << t1.get_size() << std::endl;
  // 3D tensor
  Tensor<int, 10, 10, 10> t;
  for (auto i = 0; i < 10; i++) {
    for (auto j = 0; j < 10; j++) {
      for (auto k = 0; k < 10; k++) {
        t[i][j][k] = 0.0;
      }
    }
  }
  std::cout << t.get_size() << std::endl;
  return 0;
}
