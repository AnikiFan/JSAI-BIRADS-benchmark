import torch
import torchvision
from torch import nn
from d2l import torch as d2l
class ConvolutionBlock(d2l.Classifier):
    def __init__(self,conv_num,out_channels=64,lr=0.1,first=False):
        super().__init__()
        if first:
            self.net =  nn.Sequential(
                nn.LazyConv2d(out_channels,kernel_size=7,stride=1,padding=3),
                nn.LazyBatchNorm2d()
            )
        else:
            conv_blks = []
            for _ in range(conv_num):
                conv_blks.append(nn.LazyConv2d(out_channels,kernel_size=3,stride=1,padding=1))
            self.net = nn.Sequential(
                *conv_blks,
                nn.LazyBatchNorm2d(),
                nn.MaxPool2d(kernel_size=2,stride=2) # 论文中使用的是MaxPool
            )

