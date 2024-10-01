import torch
from torch import nn


class LinearSanityChecker(nn.Module):
    def __init__(self, num_classes=10,**kwargs):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(num_classes)
        )

    def forward(self, x):
        return self.net(x)

class ConvSanityChecker(nn.Module):
    def __init__(self, num_classes=10,**kwargs):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.LazyConv2d(out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Flatten(),
            nn.LazyLinear(num_classes)
        )

    def forward(self, x):
        return self.net(x)
