/**
 * @file libtorch_test.cpp
 * @note sample for minimum executable for libtorch
 */

#include <torch/cuda.h>
#include <torch/torch.h>
#include <iostream>

int main() {
  std::cout << "=========" << std::endl;
  std::cout << "test: torch::rand({2, 3})" << std::endl;

  torch::Tensor t1 = torch::rand({2, 3});
  std::cout << t1 << std::endl;

  std::cout << std::endl << "=========" << std::endl;
  std::cout << "test: t2 = torch::rand({2, 3, 4})" << std::endl;

  torch::Tensor t2 = torch::rand({2, 3, 4});
  std::cout << t2 << std::endl;
  std::cout << t2.tensor_data() << std::endl;

  std::cout << std::endl << "=========" << std::endl;
  std::cout << "test: assign value to an element of previous tensor" << std::endl;

  t2[1][2][3] = 1234.1234;
  std::cout << t2 << std::endl;

  std::cout << std::endl << "=========" << std::endl;
  std::cout << "test: indexing, t2[0] " << std::endl;

  std::cout << t2[0] << std::endl;

  std::cout << std::endl << "=========" << std::endl;
  std::cout << "test: unsqueeze of t2" << std::endl;

  auto t2_unsq = t2.unsqueeze(0);
  std::cout << t2_unsq << std::endl;

  std::cout << std::endl << "=========" << std::endl;
  std::cout << "test: t3 = torch::zeros({4, 3, 5})" << std::endl;

  torch::Tensor t3 = torch::zeros({4, 3, 5});
  std::cout << t3 << std::endl;

  std::cout << std::endl << "=========" << std::endl;
  std::cout << "test: show t3.type(), t3.options()" << std::endl;

  std::cout << t3.type() << std::endl;
  std::cout << t3.options() << std::endl;

  std::cout << std::endl << "=========" << std::endl;
  std::cout << "test: Set t3 from CPU to GPU " << std::endl;
  std::cout << (torch::cuda::is_available() ? "cuda enable" : "cuda_disable") << std::endl;
  t3 = t3.to(torch::kCUDA);
  std::cout << t3.options() << std::endl;
  std::cout << t3 << std::endl;

  std::cout << std::endl << "=========" << std::endl;
  std::cout << "test: Make t4(zeros::({2, 3})) originally in GPU" << std::endl;
  torch::Tensor t4 = torch::zeros({2, 3}, torch::kCUDA);
  std::cout << t4 << std::endl;
  std::cout << t4.options() << std::endl;



  return 0;
}
