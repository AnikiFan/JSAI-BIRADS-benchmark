import torch
import torch.nn as nn
from torchvision import models

class MobileNetV2Classifier(nn.Module):
    def __init__(self, num_classes, feature_extract=False, pretrained=True, **kwargs):
        """
        初始化预训练的 MobileNet v2 模型，并根据需要修改分类器。

        参数:
        - num_classes (int): 分类任务的类别数。
        - feature_extract (bool): 是否冻结特征提取层。True 表示冻结，False 表示微调整个模型。
        - pretrained (bool): 是否使用预训练权重。
        """
        super(MobileNetV2Classifier, self).__init__()
        self.num_classes = num_classes
        self.feature_extract = feature_extract
        self.pretrained = pretrained

        # 加载预训练的 MobileNet v2 模型
        self.model = models.mobilenet_v2(pretrained=self.pretrained)

        if self.feature_extract:
            self.freeze_features()

        # 修改分类器以适应新的类别数
        self.modify_classifier()

    def freeze_features(self):
        """
        冻结特征提取层的参数，使其在训练过程中不更新。
        """
        for param in self.model.features.parameters():
            param.requires_grad = False

    def modify_classifier(self):
        """
        修改分类器的最后几层，以适应新的类别数。
        """
        dropout_rates = [0.5, 0.5]  # 根据需要调整 dropout 率

        # 获取特征提取部分的输出特征数
        num_ftrs = self.model.last_channel  # MobileNetV2 的最后输出通道数

        # 定义新的分类器
        self.model.classifier = nn.Sequential(
            nn.Flatten(),  # 展平成一维向量
            nn.Linear(num_ftrs, 1024),  # 全连接层 1
            nn.ReLU(),
            nn.Dropout(dropout_rates[0]),
            nn.Linear(1024, 1024),  # 全连接层 2
            nn.ReLU(),
            nn.Dropout(dropout_rates[1]),
            nn.Linear(1024, 512),  # 全连接层 3
            nn.ReLU(),
            nn.Linear(512, 128),  # 全连接层 4
            nn.ReLU(),
            nn.Linear(128, self.num_classes),  # 输出层
            # nn.Softmax(dim=1),  # 如果使用 CrossEntropyLoss，可以省略 Softmax
        )

    def forward(self, x):
        """
        前向传播函数。

        参数:
        - x (torch.Tensor): 输入图像张量，形状为 [batch_size, 3, 224, 224]

        返回:
        - torch.Tensor: 模型输出，形状为 [batch_size, num_classes]
        """
        return self.model(x)

    def get_parameter_groups(self):
        """
        获取需要优化的参数组。

        返回:
        - iterator: 需要优化的参数
        """
        if self.feature_extract:
            # 仅返回分类器的参数
            return self.model.classifier.parameters()
        else:
            # 返回所有参数
            return self.parameters()

# 示例用法
if __name__ == "__main__":
    # 设置类别数，例如 10 个类别
    num_classes = 10

    # 初始化模型
    model = MobileNetV2Classifier(num_classes=num_classes, feature_extract=True, pretrained=True)

    # 将模型移动到 GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 打印模型结构（可选）
    print(model)

    # 测试前向传播
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    outputs = model(dummy_input)
    print(f'输出形状: {outputs.shape}')  # 应该是 [1, num_classes]

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.get_parameter_groups(), lr=0.001)

    # 示例前向和反向传播
    labels = torch.tensor([1]).to(device)  # 假设目标类别为 1
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    print("模型前向和反向传播测试完成。")
