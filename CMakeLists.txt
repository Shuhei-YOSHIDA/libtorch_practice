cmake_minimum_required(VERSION 3.8)
project(libtorch_ros_practice)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# To set -std=c++17 flat to explicitly to comple_commands.json
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Get path for cmake for libtorch
#if (NOT PYTHON_EXECUTABLE)
  #set(PYTHON_EXECUTABLE "python3")
#endif()
#execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path, end='')" OUTPUT_VARIABLE pytorch_path)
#list(APPEND CMAKE_PREFIX_PATH "${pytorch_path}") # Pre-CXX11 ABI problem
#list(APPEND CMAKE_PREFIX_PATH "$ENV{HOME}/libtorch-cxx11-abi-shared-with-deps-2.4.0+cu124/libtorch/share/cmake/")
list(APPEND CMAKE_PREFIX_PATH "$ENV{HOME}/libtorch/share/cmake/")


# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(libtorch_test
  src/libtorch_test.cpp
)
target_link_libraries(libtorch_test "${TORCH_LIBRARIES}")

add_executable(detection
  src/detection.cpp
)
target_link_libraries(detection
  "${TORCH_LIBRARIES}"
  "${OpenCV_LIBS}"
)
set_property(TARGET detection PROPERTY CXX_STANDARD 17)

add_executable(detection_node
  src/detection_node.cpp
)
ament_target_dependencies(detection_node
  rclcpp
  std_msgs
  sensor_msgs
  cv_bridge
)
target_link_libraries(detection_node
  "${TORCH_LIBRARIES}"
  "${OpenCV_LIBS}"
)

install(
  TARGETS libtorch_test detection detection_node
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
