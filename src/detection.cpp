/**
 * @file detection.cpp
 * @note Use pre-learned model
 */

#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>

torch::Tensor preprocess_image(const cv::Mat& image)
{
  // 画像をリサイズ（224x224）
  cv::Mat resized_image;
  cv::resize(image, resized_image, cv::Size(224, 224));

  // BGRからRGBに変換
  cv::Mat rgb_image;
  cv::cvtColor(resized_image, rgb_image, cv::COLOR_BGR2RGB);

  // OpenCVのMatをfloat型のテンソルに変換
  torch::Tensor tensor_image = torch::from_blob(rgb_image.data, {1, 224, 224, 3}, torch::kByte);

  // チャンネルの順序を変更 (HWC -> CHW)
  tensor_image = tensor_image.permute({0, 3, 1, 2});

  // 値を[0, 1]にスケーリング
  tensor_image = tensor_image.to(torch::kFloat) / 255.0;

  return tensor_image;
}

int main(int argc, char **argv)
{
  if (argc < 3) exit(-1);
  std::string model_path(argv[1]);
  std::cout << model_path << std::endl;

  //torch::jit::script::Module model = torch::jit::load("resnet18.pt");
  torch::jit::script::Module model = torch::jit::load(model_path);

  // test
  //torch::Tensor input = torch::randn({1, 3, 224, 224});
  //torch::Tensor output = model.forward({input}).toTensor();

  std::string image_file(argv[2]);
  std::cout << image_file << std::endl;

  cv::Mat image = cv::imread(image_file, 1);
  if (image.empty())
  {
    std::cerr << "Error to load image file" << std::endl;
    return -1;
  }

  // Preprocess image and convert it into tensor
  torch::Tensor input_tensor = preprocess_image(image);

  // 推論を実行
  torch::Tensor output_tensor;
  try
  {
      output_tensor = model.forward({input_tensor}).toTensor();
  }
  catch (const c10::Error& e)
  {
      std::cerr << "エラー: 推論に失敗しました!" << std::endl;
      return -1;
  }

  // 出力を表示（仮に最初の値を表示）
  std::cout << "出力の最初の値: " << output_tensor[0][0].item<float>() << std::endl;

  return 0;
}
