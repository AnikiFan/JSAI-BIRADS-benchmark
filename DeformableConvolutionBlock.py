import torch
import torchvision
from torch import nn
from d2l import torch as d2l
class DeformableConvolutionBlock(d2l.Classifier):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=1)
        self.offset = nn.LazyConv2d(2,kernel_size=1)
        self.mask = nn.LazyConv2d(1,kernel_size=1)

    def forward(self,x):
        offset = self.offset(x)
        mask = nn.functional.sigmoid(self.mask(x))
        return torchvision.ops.deform_conv2d(input = x,offset = offset,weight=self.conv.weight,mask = mask)

