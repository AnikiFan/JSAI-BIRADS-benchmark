import torch
import torchvision
from torch import nn
from d2l import torch as d2l
from DeformableConvolutionBlock import DeformableConvolutionBlock
class DDPath(d2l.Classifier):
    def __init__(self,out_channels,kernel_size,dilation):
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyConv2d(out_channels,kernel_size=1),
            nn.LazyBatchNorm2d(),
            nn.LazyConv2d(out_channels,kernel_size=kernel_size,dilation=dilation,padding=(kernel_size//2)*dilation),
            nn.LazyBatchNorm2d()
        )

class DDModel(d2l.Classifier):
    # 指定的是单个支路的channel，总共的输出channels为7*channels
    def __init__(self,out_channels):
        super().__init__()
        self.path = []
        self.path.append( nn.Sequential(
            nn.LazyConv2d(out_channels,kernel_size=1),
            nn.LazyBatchNorm2d(),
            DeformableConvolutionBlock(out_channels,out_channels),
            nn.LazyBatchNorm2d()
        ))
        self.path.append(  nn.Sequential(
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
            nn.LazyConv2d(out_channels,kernel_size=1),
            nn.LazyBatchNorm2d()
        ))
        self.path.append( nn.Sequential(
            nn.LazyConv2d(out_channels,kernel_size=1),
            nn.LazyBatchNorm2d()
        ))
        self.path.extend([
            DDPath(out_channels,3,2),
            DDPath(out_channels,3,3),
            DDPath(out_channels,5,2),
            DDPath(out_channels,5,3)
        ])

    def forward(self,x):
        # for i,p in enumerate(self.path):
        #     print(i)
        #     p(x)
        return torch.cat(tuple(map(lambda path:path(x),self.path)),dim=1)

