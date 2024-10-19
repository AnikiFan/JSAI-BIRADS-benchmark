import torch.nn as nn
from .Block import DDBlock,IDBlock
import torch
class IDCNet(nn.Module):
    def __init__(self,num_classes,**kwargs):
        super(IDCNet, self).__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.stage1 = nn.Sequential(
            nn.LazyConv2d(kernel_size=7, stride=2, padding=3, out_channels=3), nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.MaxPool2d(stride=2, kernel_size=2),
            nn.LazyConv2d(kernel_size=7, stride=2, padding=3, out_channels=3), nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.MaxPool2d(stride=2, kernel_size=2),
        )
        self.dd1 = DDBlock(3,16)
        self.dd2 = DDBlock(16,16)
        self.dd3 = DDBlock(16,16)
        self.dd4 = DDBlock(16,16)
        self.dd5 = DDBlock(16,16)
        self.dd6 = DDBlock(16,16)
        self.dd7 = DDBlock(16,16)

        self.id1 = IDBlock()
        self.id2 = IDBlock()
        self.id3 = IDBlock()
        self.id4 = IDBlock()
        self.id5 = IDBlock()
        self.id6 = IDBlock()
        self.id7 = IDBlock()

        self.final_stage = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.LazyBatchNorm2d(),nn.Flatten(),
            nn.LazyLinear(num_classes)
        )


    def forward(self, x):
        x = self.stage1(x)

        y = self.dd1(x)
        x = torch.cat([self.id1(x),y],dim=1)

        y = self.dd2(y)
        x = torch.cat([self.id2(x),y],dim=1)

        x,y = self.maxpool(x),self.maxpool(y)

        y = self.dd3(y)
        x = torch.cat([self.id3(x), y], dim=1)

        y = self.dd4(y)
        x = torch.cat([self.id4(x), y], dim=1)

        y = self.dd5(y)
        x = torch.cat([self.id5(x), y], dim=1)

        y = self.dd6(y)
        x = torch.cat([self.id6(x), y], dim=1)

        y = self.dd7(y)
        x = torch.cat([self.id7(x), y], dim=1)
        return self.final_stage(x)

class IDNet(nn.Module):
    def __init__(self,num_classes,**kwargs):
        super(IDNet, self).__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.stage1 = nn.Sequential(
            nn.LazyConv2d(kernel_size=7,stride=2,padding=3,out_channels=3),nn.LazyBatchNorm2d(),nn.ReLU(),
            nn.MaxPool2d(stride=2,kernel_size=2),
            nn.LazyConv2d(kernel_size=7, stride=2, padding=3, out_channels=3), nn.LazyBatchNorm2d(),nn.ReLU(),
            nn.MaxPool2d(stride=2, kernel_size=2),
        )
        self.id1 = IDBlock()
        self.id2 = IDBlock()
        self.id3 = IDBlock()
        self.id4 = IDBlock()
        self.id5 = IDBlock()
        self.id6 = IDBlock()
        self.id7 = IDBlock()

        self.final_stage = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.LazyBatchNorm2d(),
            nn.Flatten(),
            nn.LazyLinear(num_classes)
        )


    def forward(self, x):
        x = self.stage1(x)
        x = self.id1(x)
        x = self.id2(x)
        x = self.maxpool(x)
        x = self.id3(x)
        x = self.id4(x)
        x = self.id5(x)
        x = self.id6(x)
        x = self.id7(x)
        x = self.final_stage(x)
        return x


