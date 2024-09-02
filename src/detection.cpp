/**
 * @file detection.cpp
 * @note Use pre-learned model of resnet18
 */

#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <tuple>

torch::Tensor preprocess_image(const cv::Mat& image)
{
  // Resize an input image（224x224）
  cv::Mat resized_image;
  cv::resize(image, resized_image, cv::Size(224, 224));

  // Convert BGR to RGB
  cv::Mat rgb_image;
  cv::cvtColor(resized_image, rgb_image, cv::COLOR_BGR2RGB);

  // Convert cv::Mat to Tensor(float)
  torch::Tensor tensor_image = torch::from_blob(rgb_image.data, {1, 224, 224, 3}, torch::kByte);

  // Change the order of channels (HWC -> CHW)
  tensor_image = tensor_image.permute({0, 3, 1, 2});

  // Scale value in between [0, 1]
  tensor_image = tensor_image.to(torch::kFloat) / 255.0;

  return tensor_image;
}

// To get high-score estimation
std::vector<int> get_top_k_indices(const torch::Tensor& output, int k)
{
  // Check shape of output tensor
  if (output.dim() != 2 || output.size(0) != 1)
  {
    std::cerr << "Storange for the shape of tensor\n";
    return {};
  }

  // Obtain score and indexes for top K
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

// Load class label from text file(Use it from pytorch repository)
std::vector<std::string> load_class_labels(const std::string& file_path)
{
  std::vector<std::string> labels;
  std::ifstream file(file_path);
  if (!file.is_open())
  {
    std::cerr << "Couldn't open text file for labels" << file_path << std::endl;
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

  // Execute inference
  torch::Tensor output_tensor;
  try
  {
    output_tensor = model.forward({input_tensor}).toTensor();
  }
  catch (const c10::Error& e)
  {
    std::cerr << "Error: Failed to infer" << std::endl;
    return -1;
  }

  // Show the output tensor information
  std::cout << "first value of tensor: " << output_tensor[0][0].item<float>() << std::endl;
  std::cout << output_tensor.sizes() << std::endl; // 1000 class in resnet
  std::cout << output_tensor.dim() << std::endl;
  
  // Get top 5 indexes
  auto top5_indices = get_top_k_indices(output_tensor, 5);

  // Show top 5 indexes and the correspoing labels
  std::cout << "top 5 indexes:[score]:[label] " << std::endl;
  for (int index : top5_indices)
  {
    std::cout << index << ":[" << output_tensor[0][index].item<float>() << "]:[" << class_labels[index] << "]" << std::endl;
  }
  std::cout << std::endl;

  return 0;
}
