# File: models/classification/resnet_classifier.py

import torch
import torch.nn as nn
from torchvision import models
from typing import Type, Any, Callable, Union, List, Optional

class ResNetClassifier(nn.Module):
    """
    一个基于 ResNet 的分类网络，支持多种 ResNet 变体（ResNet18、ResNet34、ResNet50、ResNet101、ResNet152）。
    """
    def __init__(
        self,
        resnet_type: str = 'resnet50',
        num_classes: int = 1000,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.0,
        **kwargs
    ):
        """
        初始化 ResNet 分类器。

        Args:
            resnet_type (str): 选择的 ResNet 变体名称，如 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'。
            num_classes (int): 分类任务的类别数。
            pretrained (bool): 是否加载预训练权重。
            freeze_backbone (bool): 是否冻结 ResNet 主干网络的参数。
            dropout (float): 在分类头添加 Dropout 的比例。
        """
        super(ResNetClassifier, self).__init__()
        self.resnet_type = resnet_type.lower()
        self.num_classes = num_classes

        # 根据 resnet_type 获取对应的 ResNet 模型
        self.backbone = self._get_resnet_model(self.resnet_type, pretrained)

        # 获取 ResNet 的最后一个全局平均池化层之后的特征维度
        if self.resnet_type in ['resnet18', 'resnet34']:
            self.feature_dim = 512
        elif self.resnet_type in ['resnet50', 'resnet101', 'resnet152']:
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unsupported ResNet type: {self.resnet_type}")

        # 可选的 Dropout 层
        if dropout > 0.0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

        # 替换 ResNet 的最后一个全连接层以适应 num_classes
        self.backbone.fc = nn.Linear(self.feature_dim, self.num_classes)

        # 初始化新的全连接层权重
        nn.init.kaiming_normal_(self.backbone.fc.weight, mode='fan_out', nonlinearity='relu')
        if self.backbone.fc.bias is not None:
            nn.init.constant_(self.backbone.fc.bias, 0)

        # 冻结主干网络的参数（如果需要）
        if freeze_backbone:
            self.freeze_backbone()

    def _get_resnet_model(self, resnet_type: str, pretrained: bool) -> nn.Module:
        """
        根据 resnet_type 获取对应的 ResNet 模型。

        Args:
            resnet_type (str): ResNet 变体名称。
            pretrained (bool): 是否加载预训练权重。

        Returns:
            nn.Module: 对应的 ResNet 模型。
        """
        resnet_dict = {
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50,
            'resnet101': models.resnet101,
            'resnet152': models.resnet152
        }

        if resnet_type not in resnet_dict:
            raise ValueError(f"Unsupported ResNet type: {resnet_type}. Choose from {list(resnet_dict.keys())}.")

        return resnet_dict[resnet_type](pretrained=pretrained)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, 3, H, W)。

        Returns:
            torch.Tensor: 分类输出，形状为 (batch_size, num_classes)。
        """
        features = self.backbone(x)  # ResNet 的前向传播已经包含了全连接层
        return features

    def freeze_backbone(self):
        """
        冻结 ResNet 主干网络的所有参数，使其在训练过程中不被更新。
        """
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """
        解冻 ResNet 主干网络的所有参数，使其在训练过程中可以被更新。
        """
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_backbone(self) -> nn.Module:
        """
        获取 ResNet 主干网络。

        Returns:
            nn.Module: ResNet 主干网络。
        """
        return self.backbone

    def get_classifier(self) -> nn.Module:
        """
        获取分类头（全连接层）。

        Returns:
            nn.Module: 分类头。
        """
        return self.backbone.fc

# 示例：如何使用 ResNetClassifier
if __name__ == "__main__":
    # 创建一个 ResNet50 分类器，适用于 10 类分类任务，加载预训练权重，并冻结主干网络
    model = ResNetClassifier(
        resnet_type='resnet50',
        num_classes=10,
        pretrained=True,
        freeze_backbone=True,
        dropout=0.5
    )

    # 打印模型结构
    print(model)

    # 测试前向传播
    dummy_input = torch.randn(8, 3, 224, 224)  # 假设输入图像尺寸为 224x224
    outputs = model(dummy_input)
    print(outputs.shape)  # 应输出 torch.Size([8, 10])