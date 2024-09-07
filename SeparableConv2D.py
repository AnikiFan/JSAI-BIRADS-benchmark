import torch
import torchvision
from torch import nn
from d2l import torch as d2l
class SeparableConv2D(d2l.Classifier):
    def __init__(self,in_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=in_channels,groups=in_channels,kernel_size=3,padding=1),
            nn.LazyConv2d(out_channels=1024,kernel_size=1) # 论文中没有说明该层的out_channels为多少
        )