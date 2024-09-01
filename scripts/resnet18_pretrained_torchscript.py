"""
@note To save TorchScript
@reference https://tech.anytech.co.jp/entry/2023/04/04/100000
"""

import torch
from torchvision.models import resnet18
import urllib

model = resnet18(pretrained=True)
model.eval()
dummy_input = torch.randn(1, 3, 224, 224)
# JIT compile to convert model. Invalid for converting models which changes size of input
traced_model = torch.jit.trace(model, dummy_input)
traced_model.save("resnet18.pt")

url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try:
    urllib.URLopener().retrieve(url, filename)
except:
    urllib.request.urlretrieve(url, filename)
