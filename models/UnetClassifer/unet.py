import torch
import torch.nn as nn

import sys
sys.path.append('/Users/huiyangzheng/Desktop/Project/Competition/GCAIAEC2024/AIC/TDS-Net')

from models.UnetClassifer.resnet import resnet50
from models.UnetClassifer.vgg import VGG16

# from resnet import resnet50
# from vgg import VGG16

class unetUp(nn.Module):
    # 上采样（尺寸）-> 缝合（通道）-> 卷积（通道）
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1  = nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1)
        self.conv2  = nn.Conv2d(out_size, out_size, kernel_size = 3, padding = 1)
        self.up     = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.relu   = nn.ReLU(inplace = True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs



# 编码器 解码器
class Unet(nn.Module):
    def __init__(self, in_channels ,num_classes , pretrained = False, backbone = 'vgg'):
        super(Unet, self).__init__()
        # 编码器
        if backbone == 'vgg':
            self.vgg    = VGG16(pretrained = pretrained, in_channels = in_channels)
            in_filters  = [192, 384, 768, 1024]
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained = pretrained)
            in_filters  = [192, 512, 1024, 3072]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        out_filters = [64, 128, 256, 512]

        #note: in_filters&out_filters 对应的是特征图的通道数
        
        #! upsampling
        # 64,64,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3]) #
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0]) #! 最终输出

        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor = 2), 
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None

        #! 利用卷积将channel变为num_classes
        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

        self.backbone = backbone

    def forward(self, inputs):
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        elif self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)

        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        if self.up_conv != None:
            up1 = self.up_conv(up1)

        final = self.final(up1)
        
        return final

    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True


class PretrainedClassifier(nn.Module):
    def __init__(self, num_classes=10,in_channels = 3, pretrained=False, backbone='vgg'):
        super(PretrainedClassifier, self).__init__()
        # 使用已有的Unet作为基础
        self.unet = Unet(num_classes=num_classes, in_channels=in_channels,pretrained=pretrained, backbone=backbone)
        
        # 移除解码器部分，使用Unet编码器输出
        # 假设使用VGG的最后一个特征图feat5来进行分类
        if backbone == 'vgg':
            self.feature_dim = 512  # VGG的最后一层输出通道数
        elif backbone == 'resnet50':
            self.feature_dim = 2048  # ResNet50的最后一层输出通道数
        
        # 全局平均池化层将特征图转化为特征向量
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 全连接层输出类别数
        self.fc = nn.Linear(self.feature_dim, num_classes)

    def forward(self, inputs):
        # 使用Unet的编码器提取特征
        if self.unet.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.unet.vgg.forward(inputs)
        elif self.unet.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.unet.resnet.forward(inputs)
        
        # 使用最后的特征图进行分类
        x = self.global_pool(feat5)  # 全局平均池化
        x = x.view(x.size(0), -1)    # 展平特征向量
        x = self.fc(x)               # 全连接层
        return x


# day9.25
class UnetClassifier(Unet):
    def __init__(self, in_channels, num_classes, pretrained=False, backbone='vgg'):
        super(UnetClassifier, self).__init__(in_channels, num_classes, pretrained, backbone)
        
        # 分类层
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化
        self.fc = nn.Linear(num_classes, num_classes)  # 全连接层，用于分类

    def forward(self, inputs):
        # 调用父类的前向传播
        final= super(UnetClassifier, self).forward(inputs)

        # 分类部分
        pooled = self.global_pool(final)  # 应用全局平均池化
        pooled = pooled.view(pooled.size(0), -1)  # 扁平化
        classification = self.fc(pooled)  # 分类输出

        # return classification,final # 返回分类结果和分割结果
        return classification

