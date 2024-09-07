import torch
import torchvision
from torch import nn
from d2l import torch as d2l
class FeatureBlock(d2l.Classifier):
    def __init__(self,out_channels):
        super().__init__()
        self.net =nn.Sequential(
            nn.LazyConv2d(out_channels,kernel_size=7,stride=2,padding=3),
            nn.LazyBatchNorm2d(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )