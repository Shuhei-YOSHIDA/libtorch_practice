/**
 * @file detection.cpp
 * @note Use pre-learned model
 */

#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <tuple>

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

// 上位Kのインデックスを取得する関数
std::vector<int> get_top_k_indices(const torch::Tensor& output, int k)
{
  // 出力テンソルの次元を確認
  if (output.dim() != 2 || output.size(0) != 1)
  {
    std::cerr << "テンソルの次元が期待と異なります。\n";
    return {};
  }

  // 上位Kのスコアとインデックスを取得
  auto topk_result = torch::topk(output, k, /*dim=*/1, /*largest=*/true, /*sorted=*/true);
  auto topk_values = std::get<0>(topk_result);
  auto topk_indices = std::get<1>(topk_result);

  std::vector<int> indices;
  for (int i = 0; i < k; ++i)
  {
    indices.push_back(topk_indices[0][i].item<int>());
  }

  return indices;
}

// ラベルのマッピングを読み込む関数
std::vector<std::string> load_class_labels(const std::string& file_path)
{
  std::vector<std::string> labels;
  std::ifstream file(file_path);
  if (!file.is_open())
  {
    std::cerr << "ラベルファイルを開けませんでした: " << file_path << std::endl;
    exit(-1);
  }

  std::string line;
  while (std::getline(file, line))
  {
    labels.push_back(line);
  }
  return labels;
}

int main(int argc, char **argv)
{
  if (argc < 4) exit(-1);
  std::string model_path(argv[1]);
  std::cout << model_path << std::endl;

  std::string image_file(argv[2]);
  std::cout << image_file << std::endl;

  std::string label_file(argv[3]);
  std::cout << label_file << std::endl;

  auto class_labels = load_class_labels(label_file);
  std::cout << class_labels.size() << std::endl;

  //torch::jit::script::Module model = torch::jit::load("resnet18.pt");
  torch::jit::script::Module model = torch::jit::load(model_path);
  //model.eval();

  // test
  //torch::Tensor input = torch::randn({1, 3, 224, 224});
  //torch::Tensor output = model.forward({input}).toTensor();


  cv::Mat image = cv::imread(image_file);
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
  std::cout << output_tensor.sizes() << std::endl; // 1000 class in resnet
  std::cout << output_tensor.dim() << std::endl;
  
  // 上位5のインデックスを取得
  auto top5_indices = get_top_k_indices(output_tensor, 5);

  // 上位5のインデックスを表示
  std::cout << "上位5のインデックス:[score]:[label] " << std::endl;
  for (int index : top5_indices)
  {
    std::cout << index << ":[" << output_tensor[0][index].item<float>() << "]:[" << class_labels[index] << "]" << std::endl;
  }
  std::cout << std::endl;

  return 0;
}
