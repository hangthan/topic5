import torch
from torchvision import models

vgg = models.vgg16_bn()
features = list(vgg.features.children())
print(f"Total features: {len(features)}")
for i, f in enumerate(features):
    name = type(f).__name__
    extra = ""
    if hasattr(f, "in_channels"):
        extra = f"in={f.in_channels}, out={f.out_channels}"
    elif hasattr(f, "num_features"):
        extra = f"features={f.num_features}"
    elif name == "MaxPool2d":
        extra = f"kernel={f.kernel_size}, stride={f.stride}"
    print(f"  [{i:2d}] {name:15s} {extra}")
