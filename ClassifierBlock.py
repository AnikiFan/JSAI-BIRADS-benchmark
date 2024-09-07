import torch
import torchvision
from torch import nn
from d2l import torch as d2l
from SeparableConv2D import SeparableConv2D
from SELayer import SELayer
class ClassifierBlock(d2l.Classifier):
    def __init__(self,num_class,in_channels):
        super().__init__()
        self.net = nn.Sequential(
            SeparableConv2D(in_channels),
            SELayer(1024),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.LazyBatchNorm1d(),
            nn.LazyLinear(num_class)
        )