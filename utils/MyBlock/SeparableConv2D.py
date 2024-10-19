from torch import nn
import torch.cuda


class SeparableConv2D(nn.Module):
    def __init__(self, in_channels,out_channels=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, groups=in_channels, kernel_size=3, padding=1),nn.ReLU(),
            nn.LazyConv2d(out_channels=out_channels, kernel_size=1)  # 论文中没有说明该层的out_channels为多少
        )
        if torch.cuda.is_available():
            self.net = self.net.to(torch.device('cuda'))

    def forward(self, x):
        return self.net(x)
