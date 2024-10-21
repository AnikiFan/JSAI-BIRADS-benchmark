from .model_v2 import MobileNetV2
from torch import nn
import torch
import os


class MyMobileNetV2(nn.Module):
    def __init__(self, model_weight_path,pretrained=True, **kwargs):
        super().__init__()
        model_weight_path = os.path.join(model_weight_path, "mobilenet_v2_pre.pth")
        assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
        # 创建模型并移动到目标设备
        model = MobileNetV2()  # 假设你的模型是 MobileNetV2
        # 加载预训练权重
        pretrained_dict = torch.load(model_weight_path,weights_only=True)
        model_dict = model.state_dict()
        # 过滤掉全连接层的权重
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and "fc" not in k}
        # 更新模型字典
        model_dict.update(pretrained_dict)
        # 加载过滤后的参数
        model.load_state_dict(model_dict)
        #######################################3四个分支全连接层
        model.fc1[1] = torch.nn.Linear(model.fc1[1].in_features, 1)
        model.fc2[1] = torch.nn.Linear(model.fc2[1].in_features, 1)
        model.fc3[1] = torch.nn.Linear(model.fc3[1].in_features, 1)
        model.fc4[1] = torch.nn.Linear(model.fc4[1].in_features, 1)  # 根据需要修改输出层
        self.net = model

    def forward(self, x):
        return self.net(x)
    
    

