import torch.cuda
from torch import nn
from utils.MyBlock.SeparableConv2D import SeparableConv2D
from utils.MyBlock.SELayer import SELayer


class ClassifierBlock(nn.Module):
    def __init__(self, num_class, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            SeparableConv2D(in_channels),
            SELayer(1024),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LazyBatchNorm1d(),
            nn.LazyLinear(num_class)
        )
        if torch.cuda.is_available():
            self.net = self.net.to(torch.device('cuda'))

    def forward(self, x):
        return self.net(x)
