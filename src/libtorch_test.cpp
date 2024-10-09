/**
 * @file libtorch_test.cpp
 * @note sample for minimum executable for libtorch
 */

#include <torch/cuda.h>
#include <torch/torch.h>
#include <iostream>
#include <random>
#include <chrono>

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

  std::cout << std::endl << "=========" << std::endl;
  std::cout << "test: naive assignment" << std::endl;
  const int height = 32;
  const int width = 1800;
  //std::random_device seed_gen;
  std::mt19937 engine(0);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<std::vector<float>> v;
  v.reserve(height);
  for (int h = 0; h < height; h++)
  {
    std::vector<float> vv;
    vv.reserve(width);
    for (int w = 0; w < width; w++)
    {
      vv.push_back(dist(engine));
    }
    v.push_back(vv);
  }
  torch::Tensor t5 = torch::zeros({height, width});
  auto start = std::chrono::system_clock::now();
  for (int h = 0; h < height; h++)
  {
    for (int w = 0; w < width; w++)
    {
      t5[h][w] = v[h][w];
    }
  }
  auto end = std::chrono::system_clock::now();
  // Too slow
  double elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
  std::cout << "h/w elappsed time[ms]: " << height << "/" << width << " " << elapsed_ms << " ms" << std::endl;

  std::cout << std::endl << "=========" << std::endl;
  std::cout << "test: from_blob" << std::endl;
  std::vector<float> d = {0., 1., 2., 3., 4., 5.};
  torch::Tensor t6 = torch::from_blob(d.data(), {2, 3});
  torch::Tensor t7 = torch::from_blob(d.data(), {3, 2});
  std::cout << t6 << std::endl;
  std::cout << t7 << std::endl;

  std::cout << std::endl << "=========" << std::endl;
  std::cout << "test: from_blob and long data" << std::endl;

  start = std::chrono::system_clock::now();
  std::vector<float> oneline_v;
  oneline_v.reserve(height*width);
  for (int h = 0; h < height; h++)
  {
    auto h_line = v[h];
    for (int w = 0; w < width; w++)
    {
      oneline_v.push_back(h_line[w]);
    }
  }
  torch::Tensor t8 = torch::from_blob(oneline_v.data(), {height, width});
  end = std::chrono::system_clock::now();
  elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
  // Fast tensor initialization
  std::cout << "elapsed time: " << elapsed_ms << "micro-sec for " << t8.sizes() << std::endl;


  return 0;
}
