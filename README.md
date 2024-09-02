libtorch_ros_practice
====

## How to use
* Use libtorch for CXX11 ABI not pre-CXX11 ABI
* libtorch installed with pytorch by pip is pre-CXX11 ABI
* Download libtorch to your home

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
