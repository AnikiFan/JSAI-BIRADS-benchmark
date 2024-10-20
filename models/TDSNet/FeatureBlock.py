from torch import nn
import torch.cuda


class FeatureBlock(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyConv2d(out_channels, kernel_size=7, stride=2, padding=3),nn.LazyBatchNorm2d(),nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        if torch.cuda.is_available():
            self.net = self.net.to(torch.device('cuda'))

    def forward(self, x):
        return self.net(x)
