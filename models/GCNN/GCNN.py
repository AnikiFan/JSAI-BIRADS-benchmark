import torch.nn as nn
import torch
class GCNN(nn.Module):
    def __init__(self,num_classes,**kwargs):
        super(GCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.LazyConv2d(64,kernel_size=3,padding=1),nn.ReLU(),
            nn.LazyConv2d(64,kernel_size=3,padding=1),nn.ReLU(),
            nn.MaxPool2d(2,stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.LazyConv2d(128, kernel_size=3, padding=1), nn.ReLU(),
            nn.LazyConv2d(128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.LazyConv2d(256, kernel_size=3, padding=1), nn.ReLU(),
            nn.LazyConv2d(256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        self.layer4 = nn.Sequential(
            nn.LazyConv2d(512, kernel_size=3, padding=1), nn.ReLU(),
            nn.LazyConv2d(512, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        self.layer5 = nn.Sequential(
            nn.LazyConv2d(1024, kernel_size=3, padding=1), nn.ReLU(),
            nn.LazyConv2d(1024, kernel_size=3, padding=1), nn.ReLU(),
            nn.LazyConv2d(1024, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.bypath1 = nn.Sequential(
            nn.LazyConv2d(64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(8, stride=8)
        )
        self.bypath2 = nn.Sequential(
            nn.LazyConv2d(128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(4, stride=4)
        )
        self.bypath3 = nn.Sequential(
            nn.LazyConv2d(256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        self.catconv1 = nn.LazyConv2d(1024, kernel_size=3, padding=1)
        self.catconv2 = nn.LazyConv2d(1024, kernel_size=3, padding=1)
        self.catconv3 = nn.LazyConv2d(1024, kernel_size=3, padding=1)
        self.catconv4 = nn.LazyConv2d(1024, kernel_size=3, padding=1)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),nn.Flatten(),
            nn.LazyLinear(4096), nn.ReLU(),
            nn.LazyLinear(1024), nn.ReLU(),
            nn.LazyLinear(num_classes)
        )

        self.relu = nn.ReLU()


    def forward(self,x):
        x = self.layer1(x)
        y1 =  self.bypath1(x)

        x = self.layer2(x)
        y2 = self.bypath2(x)

        x = self.layer3(x)
        y3 = self.bypath3(x)

        x = self.layer4(x)
        y4 = x

        x = self.layer5(x)

        x = torch.cat([x,y4],dim=1)
        x = self.catconv1(x)
        x = self.relu(x)

        x = torch.cat([x,y3],dim=1)
        x = self.catconv2(x)
        x = self.relu(x)

        x = torch.cat([x,y2],dim=1)
        x = self.catconv3(x)
        x = self.relu(x)

        x = torch.cat([x,y1],dim=1)
        x = self.catconv4(x)
        x = self.relu(x)

        return self.classifier(x)





