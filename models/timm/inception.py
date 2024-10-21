import torch
import torch.nn as nn
import timm
import logging
from logging import info
from torchsummary import summary


class InceptionClassifier(nn.Module):
    def __init__(
        self,
        model_name="inception_v3.ra_in1k", 
        pretrained=True, 
        num_classes=6, 
        features_only=True,
        freeze_backbone=True,
        **kwargs
    ):
        super(InceptionClassifier, self).__init__()
        pathToCheckpoints = {
            # "densenet121.ra_in1k": "./model_data/densenet121.ra_in1k.bin",
            'inception_resnet_v2.tf_in1k': "./model_data/inception_resnet_v2.tf_in1k.bin",
        }
        # 模型参数
        self.features_only = features_only
        self.freeze_backbone = freeze_backbone
        self.num_classes = num_classes
        
        if features_only:
            print(f"mode:Feature Extraction")
            self.base_model = timm.create_model(
            model_name, 
            pretrained=pretrained,
            pretrained_cfg_overlay=dict(file=pathToCheckpoints[model_name]),
            features_only=features_only,
            )
        elif num_classes == 0:
            print(f"mode:Image Embedding")
            self.base_model = timm.create_model(
            model_name, 
            pretrained=pretrained,
            pretrained_cfg_overlay=dict(file=pathToCheckpoints[model_name]),
            num_classes=num_classes
            )
        elif num_classes > 0:
            print(f"mode:Image Classification")
            self.base_model = timm.create_model(
            model_name, 
            pretrained=pretrained,
            pretrained_cfg_overlay=dict(file=pathToCheckpoints[model_name]),
            num_classes=num_classes
            )
        else:
            raise ValueError(f"num_classes must be greater than 0, but got {num_classes}")
            
        # print(f"base_model: {self.base_model}")
        
        # 冻结backbone
        # if freeze_backbone:
        #     for param in self.base_model.parameters():
        #         param.requires_grad = False
                
        # 根据模型获取transform
        self.train_transform, self.val_transform = self.get_transform()
        
        # 分类器来自github项目的推荐
        dropout_rates = (0.5, 0.3)
        # # 获取模型的输出特征维度
        # if hasattr(self.base_model, 'num_features'):
        #     num_features = self.base_model.num_features
        # elif hasattr(self.base_model, 'fc'):
        #     num_features = self.base_model.fc.in_features
        
        self.classifier = nn.Sequential(
            nn.Flatten(),  # 展平特征
            nn.Linear(16, 1024),  # Dense Layer 1
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
            # nn.Softmax(dim=1),  # Softmax 激活函数
        )
        
    def get_transform(self):
        '''
        获取模型的transform
        '''
        # 获取模型的数据配置
        data_config = timm.data.resolve_model_data_config(self.base_model)
        print(data_config)
        
        train_transform = timm.data.create_transform(**data_config, is_training=True)
        val_transform = timm.data.create_transform(**data_config, is_training=False)
        
        # 返回训练和验证的数据转换
        return train_transform, val_transform
    

    def forward(self, x):
        '''
        前向传播
        '''
        # 根据模型是否在训练，选择使用训练的transform还是验证的transform
        # if self.training:
        #     x = self.train_transform(x)
        # else:
        #     x = self.val_transform(x)
        
        # 前向传播，backbone输出特征
        x = self.base_model(x)
        # 如果基模型输出的是列表或元组（例如，多个特征层），选择最后一个特征层
        # if isinstance(x, (list, tuple)):
        #     x = x[-1]
        
        # 分类器输出分类结果
        # x = self.classifier(x)
        return x

    
    # def get_features(self, x):
    #     '''
    #     获取backbone的输出特征
    #     '''
    #     # 根据模型是否在训练，选择使用训练的transform还是验证的transform
    #     if self.training:
    #         x = self.train_transform(x)
    
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = InceptionClassifier(model_name="inception_resnet_v2.tf_in1k", pretrained=True, num_classes=6, features_only=True, freeze_backbone=True)
    model.to(device)  # 将模型移动到指定设备
    
    # 创建一个示例输入并移动到指定设备
    example_input = torch.randn(20, 3, 224, 224).to(device)  # 假设输入是224x224的RGB图像 batch, channel, height, width

    # 打印模型摘要
    summary(model, input_size=(3, 224, 224), device=str(device))
    
    # 获取模型最后一层的输出
    with torch.no_grad():
        features = model(example_input)
    
    # 打印特征的形状
    print("最后一层输出的特征形状:", features.shape)
    
    # 如果想查看具体的特征值，可以取其中的一部分打印出来
    print("特征值示例（前10个）:", features[0, :10])
    
