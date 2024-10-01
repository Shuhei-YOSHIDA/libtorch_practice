/**
 * @file detection_component.cpp
 */

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/utilities.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>

#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <tuple>

using std::placeholders::_1;

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

class DetectionComp : public rclcpp::Node
{
public:
  DetectionComp(const rclcpp::NodeOptions& options)
    : Node("detection_component", options)
  {
    _img_sub = this->create_subscription<sensor_msgs::msg::Image>(
        "image_raw", 1, std::bind(&DetectionComp::img_cb, this, _1));

    _annotated_img_pub = this->create_publisher<sensor_msgs::msg::Image>("annotated_img", 1);

    // Get parameter
    std::string model_path = this->declare_parameter("model_path", "");
    std::string label_path = this->declare_parameter("label_path", "");

    // Prepare libtorch-related
    try
    {
      _model = torch::jit::load(model_path);
      _class_labels = load_class_labels(label_path);
    }
    catch(std::exception& e)
    {
      RCLCPP_ERROR(this->get_logger(), "Failed to load model_path or label_path: %s", e.what());
      rclcpp::shutdown();
      return; 
    }
  }

private:
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr _annotated_img_pub;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr _img_sub;
  void img_cb(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    //auto img = cv_bridge::toCvShare(msg, msg->encoding);
    auto img = cv_bridge::toCvShare(msg, "bgr8");
    if (img->image.empty()) return;

    torch::Tensor input_tensor = preprocess_image(img->image);

    torch::Tensor output_tensor;
    try
    {
      output_tensor = _model.forward({input_tensor}).toTensor();
    }
    catch(const c10::Error& e)
    {
      std::cerr << "Error: Failed to infer" << std::endl;
      return;
    }

    // Get top 5 indexes
    auto top5_indices = get_top_k_indices(output_tensor, 5);


    int cnt = 1;
    for (int index : top5_indices )
    {
      cv::putText(img->image, _class_labels[index], cv::Point(20, 20*cnt), cv::FONT_HERSHEY_SIMPLEX, 1.0, CV_RGB(255, 0, 0), 2);
      cnt++;
    }

    sensor_msgs::msg::Image output_msg;
    img->toImageMsg(output_msg);

    _annotated_img_pub->publish(output_msg);

  }

  torch::jit::script::Module _model;
  std::vector<std::string> _class_labels;
};

RCLCPP_COMPONENTS_REGISTER_NODE(DetectionComp)
