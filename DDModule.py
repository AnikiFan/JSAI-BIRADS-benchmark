import torch
from torch import nn
from DeformableConvolutionBlock import DeformableConvolutionBlock


class DDPath(nn.Module):
    def __init__(self, out_channels, kernel_size, dilation):
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyConv2d(out_channels, kernel_size=1),
            nn.LazyBatchNorm2d(),
            nn.LazyConv2d(out_channels, kernel_size=kernel_size, dilation=dilation,
                          padding=(kernel_size // 2) * dilation),
            nn.LazyBatchNorm2d()
        )
        if torch.cuda.is_available():
            self.net = self.net.to(torch.device('cuda'))

    def forward(self, x):
        return self.net(x)


class DDModel(nn.Module):
    # 指定的是单个支路的channel，总共的输出channels为7*channels
    def __init__(self, out_channels):
        super().__init__()
        self.path = []  # 为了调试可能顺序与论文中有差异
        self.path.append(nn.Sequential(  # 1
            nn.LazyConv2d(out_channels, kernel_size=1),
            nn.LazyBatchNorm2d(),
            DeformableConvolutionBlock(out_channels, out_channels),
            nn.LazyBatchNorm2d()
        ))
        self.path.append(nn.Sequential(  # 2
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.LazyConv2d(out_channels, kernel_size=1),
            nn.LazyBatchNorm2d()
        ))
        self.path.append(nn.Sequential(  # 3
            nn.LazyConv2d(out_channels, kernel_size=1),
            nn.LazyBatchNorm2d()
        ))
        if torch.cuda.is_available():
            for i, p in enumerate(self.path):
                self.path[i] = p.to(torch.device('cuda'))
        self.path.extend([  # 4-7
            DDPath(out_channels, 3, 2),
            DDPath(out_channels, 3, 3),
            DDPath(out_channels, 5, 2),
            DDPath(out_channels, 5, 3)
        ])

    def forward(self, x):
        # res = []
        # for i,p in enumerate(self.path):
        #     print(i)
        #     res.append(p(x))
        # return torch.cat(res,dim = 1)
        return torch.cat(tuple(map(lambda path: path(x), self.path)), dim=1)
