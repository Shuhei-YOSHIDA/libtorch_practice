libtorch_ros_practice
====

## How to use
* Use libtorch for CXX11 ABI not pre-CXX11 ABI
* libtorch installed with pytorch by pip is pre-CXX11 ABI
* Download libtorch to your home
* Set `CMAKE_PREFIX_PATH` to libtorch directory you download in `CMakeLists.txt`.


For test by one image,
```
$ cd ~/ros2ws/src/libtorch_ros_practice
$ python3 scripts/resnet18_pretrained_torchscript.py
$ ros2 run libtorch_ros_practice detection resnet18.pt dog.jpg imagenet_classes.txt
resnet18.pt
dog.jpg
imagenet_classes.txt
1000
first value of tensor: -1.00588
[1, 1000]
2
top 5 indexes:[score]:[label]
258:[14.0042]:[Samoyed]
261:[10.5235]:[keeshond]
279:[10.2586]:[Arctic fox]
259:[9.82442]:[Pomeranian]
283:[9.69924]:[Persian cat]
```

To start ROS node,
```bash
$ ros2 run usb_cam usb_cam_node_exe # start publishing /image_raw
$ cd ~/ros2ws/src/libtorch_ros_practice
$ ros2 run libtorch_ros_practice detection_node resnet18.pt imagenet_classes.txt
```
[![example](http://img.youtube.com/vi/n00RjAHWl34/0.jpg)](https://www.youtube.com/watch?v=n00RjAHWl34)

To use ROS components, you need to set environment variable `LD_LIBRARY_PATH`.
```bash
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH${HOME}/libtorch/lib # example
$ ros2 run libtorhc_ros_practice detection_component_node --ros-args -p model_path:=resnet18.pt label_path:=imagenet_classes.txt
```
