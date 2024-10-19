import torch.nn as nn
from utils.MyBlock.SeparableConv2D import SeparableConv2D
import torch
class IDBlock(nn.Module):
    def __init__(self):
        super(IDBlock, self).__init__()
        self.b1 = nn.Sequential(
            nn.LazyConv2d(kernel_size=1,out_channels=32), nn.LazyBatchNorm2d(),nn.ReLU(),
        )
        self.b2 = self._make_branch(3,1,32)
        self.b3 = self._make_branch(3,2,32)
        self.b4 = self._make_branch(5,1,32)
        self.b5 = self._make_branch(5,2,32)
        self.b6 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
            nn.LazyConv2d(kernel_size=1,out_channels=32),nn.LazyBatchNorm2d(),nn.ReLU(),
        )

    @staticmethod
    def _make_branch(k,r,out_channels):
        return nn.Sequential(
            nn.LazyConv2d(out_channels=out_channels,kernel_size=1),nn.LazyBatchNorm2d(),nn.ReLU(),
            nn.LazyConv2d(kernel_size=k,dilation=r,out_channels=out_channels,padding=(k//2)*r),nn.LazyBatchNorm2d(),nn.ReLU(),
        )

    def forward(self, x):
        return torch.cat([self.b1(x),self.b2(x),self.b3(x),self.b4(x),self.b5(x),self.b6(x)],1)

class DDBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(DDBlock,self).__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.LazyBatchNorm2d(),
            SeparableConv2D(in_channels=in_channels,out_channels=out_channels),nn.ReLU(),
            nn.LazyConv2d(kernel_size=3,dilation=2,out_channels=out_channels,padding=2),nn.ReLU(),
        )

    def forward(self,x):
        return self.net(x)



