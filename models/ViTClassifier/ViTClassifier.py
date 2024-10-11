import torch
import torch.nn as nn
import timm

class ViTClassifier(nn.Module):
    def __init__(self, 
                 model_name='vit_base_patch16_224', 
                 pretrained=True, 
                 num_classes=10, 
                 drop_rate=0.0, 
                 drop_path_rate=0.1,
                 **kwargs):
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

        super(ViTClassifier, self).__init__()
        
        pathToCheckpoints = {
        #     'vit_base_patch16_224': 'https://download.pytorch.org/models/vit_base_patch16_224-b5f2ef4d.pth',
            'vit_base_patch16_224': './model_data/vit_base_patch16_224.bin'
        }
        
        # 使用 timm 创建预训练的 ViT 模型
        self.vit = timm.create_model(model_name, 
                                     pretrained=pretrained, 
                                     pretrained_cfg_overlay=dict(file=pathToCheckpoints[model_name]),
                                     num_classes=num_classes, 
                                     drop_rate=drop_rate, 
                                    drop_path_rate=drop_path_rate)
        
        # 如果需要自定义分类头，可以取消注释以下部分
        # base_model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        # self.vit = base_model
        # self.classifier = nn.Sequential(
        #     nn.Dropout(drop_rate),
        #     nn.Linear(base_model.num_features, num_classes)
        # )
        data_config = timm.data.resolve_model_data_config(self.vit)
        print(data_config)
        self.transform = timm.data.create_transform(**data_config, is_training=True)
        print(self.transform)

    def forward(self, x):
        """
        前向传播

        参数：
        - x (torch.Tensor): 输入图像张量，形状为 (batch_size, 3, 224, 224)
        
        返回：
        - torch.Tensor: 分类结果，形状为 (batch_size, num_classes)
        """
        return self.vit(x)
        # 如果自定义了分类头，请使用以下代码
        # x = self.vit(x)
        # x = self.classifier(x)
        # return x


# 示例用法
if __name__ == "__main__":
    # 假设有10个类别
    num_classes = 10
    # 选择你需要的模型名称
    model = ViTClassifier(model_name='vit_base_patch16_224', 
                          pretrained=True, 
                          num_classes=num_classes)
    
    # 打印模型结构
    print(model)
    
    # 测试前向传播
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(output.shape)  # 应输出 torch.Size([1, 10])