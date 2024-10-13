import torch
import torch.nn as nn
import timm
from logging import info


# class ViTClassifier(nn.Module):
class ViTClassifier_timm(nn.Module):
    def __init__(
        self,
        model_name="vit_base_patch16_224",
        pretrained=True,
        num_classes=10,
        drop_rate=0.0,
        drop_path_rate=0.1,
        freeze_layers=0,
        classifier_head=False,
        **kwargs,
    ):
        """
        Vision Transformer 分类器

        参数：
        - model_name (str): timm 库中 Vision Transformer 模型的名称。
        - pretrained (bool): 是否加载预训练权重。
        - num_classes (int): 分类的类别数量。
        - drop_rate (float): Dropout 概率，用于正则化。
        - drop_path_rate (float): Drop path 概率，用于正则化。

        可选的模型名称包括但不限于：
        - 'vit_base_patch16_224'：基础版 ViT，补丁大小 16x16，输入尺寸 224x224，参数约 86M。
        - 'vit_large_patch16_224'：大型 ViT，补丁大小 16x16，输入尺寸 224x224，参数约 307M。
        - 'vit_small_patch16_224'：小型 ViT，补丁大小 16x16，输入尺寸 224x224，参数约 22M。
        - 'vit_tiny_patch16_224'：微型 ViT，补丁大小 16x16，输入尺寸 224x224，参数约 5M。
        - 'deit_base_patch16_224'：Data-efficient Image Transformer (DeiT) 基础版，补丁大小 16x16，输入尺寸 224x224，参数约 86M。
        - 'deit_small_patch16_224'：DeiT 小型版，补丁大小 16x16，输入尺寸 224x224，参数约 22M。
        - 'vit_base_patch32_224'：基础版 ViT，补丁大小 32x32，输入尺寸 224x224。
        - 'vit_base_patch16_384'：基础版 ViT，补丁大小 16x16，输入尺寸 384x384。
        - 'vit_base_resnet50d'：混合架构，结合 ResNet 和 ViT 的特性。
        - 'vit_base_miil'：多任务学习版本，适用于多任务学习场景。

        你可以根据以下因素选择合适的模型：
        1. **模型规模**：选择 `vit_tiny_patch16_224` 或 `vit_small_patch16_224` 适用于计算资源有限或需要快速推理的场景；选择 `vit_base_patch16_224` 或 `vit_large_patch16_224` 适用于需要高精度且计算资源充足的场景。
        2. **输入图像尺寸**：标准输入尺寸为 224x224，如果需要更高分辨率，可以选择支持 384x384 的模型，如 `vit_base_patch16_384`。
        3. **数据集规模**：对于大规模数据集（如 ImageNet），大型模型通常表现更好；对于小规模数据集，轻量级模型可能更适合，并且可以更快地训练。
        4. **特定任务需求**：如需要结合其他架构的特性，可以选择 `vit_base_resnet50d` 等混合架构模型。

        例如：
        ```python
        # 示例：使用小型 ViT 模型
        model = ViTClassifier(model_name='vit_small_patch16_224',
                              pretrained=True,
                              num_classes=10)
        ```
        """
        model_config = {
            "vit_base_patch16_224":{
                "crea"
                "checkpoint_path":"./model_data/vit_base_patch16_224.bin"
            }
            
        }
        super(ViTClassifier_timm, self).__init__()

        pathToCheckpoints = {
            #     'vit_base_patch16_224': 'https://download.pytorch.org/models/vit_base_patch16_224-b5f2ef4d.pth',
            "vit_base_patch16_224": "./model_data/vit_base_patch16_224.bin",
            "vit_mediumd_patch16_reg4_gap_256.sbb_in12k_ft_in1k": "./model_data/vit_mediumd_patch16_reg4_gap_256.sbb_in12k_ft_in1k.bin",
            "fastvit_ma36.apple_in1k": "./model_data/fastvit_ma36.apple_in1k.bin",
            "densenet121.ra_in1k": "./model_data/densenet121.ra_in1k.bin",
        }

        # 使用 timm 创建预训练的 ViT 模型
        self.base_model = timm.create_model(
            model_name,
            pretrained=pretrained,
            pretrained_cfg_overlay=dict(file=pathToCheckpoints[model_name]),
            num_classes=num_classes if not classifier_head else 0,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )
        for param in self.base_model.parameters():
            param.requires_grad = True

        # 如果需要自定义分类头，可以取消注释以下部分
        # base_model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.classifier_head = classifier_head
        if self.classifier_head:
            # self.classifier = nn.Sequential(
            #     nn.Dropout(drop_rate),
            #     nn.Linear(self.base_model.num_features, 256),
            #     nn.BatchNorm1d(256),
            #     nn.ReLU(),
            #     nn.Dropout(drop_rate),
            #     nn.Linear(256, num_classes)
            # )
            dropout_rates = (0.5, 0.3)
            self.classifier = nn.Sequential(
                nn.Flatten(),  # 展平特征
                nn.Linear(self.base_model.num_features, 1024),  # Dense Layer 1
                nn.ReLU(),
                nn.Dropout(dropout_rates[0]),
                nn.Linear(1024, 1024),  # Dense Layer 2
                nn.ReLU(),
                nn.Dropout(dropout_rates[1]),
                nn.Linear(1024, 512),  # Dense Layer 3
                nn.ReLU(),
                nn.Linear(512, 128),  # Dense Layer 4
                nn.ReLU(),
                nn.Linear(128, num_classes),  # Dense Output Layer
                nn.Softmax(dim=1),  # Softmax 激活函数
            )

        # 冻结前 freeze_layers 层
        if freeze_layers > 0:
            for name, param in self.base_model.named_parameters():
                if int(name.split(".")[1]) < freeze_layers:
                    param.requires_grad = False

        # 打印模型信息
        data_config = timm.data.resolve_model_data_config(self.base_model)
        self.transform = timm.data.create_transform(**data_config, is_training=True)
        info(f"------------------------------------------------------------------")
        info(f"info about {model_name}")
        info(f"数据配置:\n{data_config}")
        info(f"预训练模型所用变换:\n{self.transform}")
        info(f"变换类型:     {type(self.transform)}")
        info(f"------------------------------------------------------------------")

    def forward(self, x):
        """
        前向传播

        参数：
        - x (torch.Tensor): 输入图像张量，形状为 (batch_size, 3, 224, 224)

        返回：
        - torch.Tensor: 分类结果，形状为 (batch_size, num_classes)
        """
        # return self.vit(x)
        # 如果自定义了分类头，请使用以下代码
        x = self.base_model(x)
        if self.classifier_head:
            x = self.classifier(x)
        return x


# 示例用法
if __name__ == "__main__":
    # 假设有10个类别
    num_classes = 10
    # 选择你需要的模型名称
    model = ViTClassifier(
        model_name="vit_base_patch16_224", pretrained=True, num_classes=num_classes
    )

    # 打印模型结构
    print(model)

    # 测试前向传播
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(output.shape)  # 应输出 torch.Size([1, 10])
