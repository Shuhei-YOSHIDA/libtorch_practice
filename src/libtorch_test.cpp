/**
 * @file libtorch_test.cpp
 * @note sample for minimum executable for libtorch
 */

#include <torch/torch.h>
#include <iostream>

int main() {
  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << tensor << std::endl;
}
