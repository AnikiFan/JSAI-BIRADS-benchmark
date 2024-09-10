import torch
from torch import nn
from FeatureBlock import FeatureBlock
from ClassifierBlock import ClassifierBlock
from ConvolutionBlock import ConvolutionBlock
from DDModule import DDModel


class TDSNet(nn.Module):
    def __init__(self, num_class=10):
        super().__init__()

        self.featureBlock = FeatureBlock(64)  # out_channels = 64

        self.classifierBlock = ClassifierBlock(num_class, 1024)

        self.cb1 = ConvolutionBlock(1, 64, first=True)  # out_channels = 64
        self.cb2 = ConvolutionBlock(2, 128)  # out_channels = 128
        self.cb3 = ConvolutionBlock(2, 128)  # out_channels = 128
        self.cb4 = ConvolutionBlock(2, 128)  # out_channels = 128

        self.db1 = DDModel(64)
        self.db2 = nn.Sequential(
            DDModel(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            DDModel(128)
        )
        self.db3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DDModel(128)
        )
        self.db4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DDModel(128)
        )

    def forward(self, x):
        x = self.featureBlock(x)
        y = self.cb1(x)
        x = self.db1(x)
        x = torch.concat((x, y), dim=1)
        y = self.cb2(y)
        x = self.db2(x)
        x = torch.concat((x, y), dim=1)
        y = self.cb3(y)
        x = self.db3(x)
        x = torch.concat((x, y), dim=1)
        y = self.cb4(y)
        x = self.db4(x)
        x = torch.concat((x, y), dim=1)
        return self.classifierBlock(x)
