from torch.nn import BCELoss, Sigmoid
from torch import Tensor
import torch
import torch.nn as nn
from typing import *
from utils.tools import getDevice
from torchvision.ops import sigmoid_focal_loss

class MyBCELoss:
    def __int__(self):
        """
        专门针对fea实现的BCELoss，fea任务所用模型的输出可能含有负数，所以需要使用Sigmoid
        :param input: 
        :param target: 
        :return: 
        """
        pass

    def __call__(self, input, target, **kwargs):
        return BCELoss()(Sigmoid()(input), target)


class WeightedCrossEntropyLoss(nn.Module):
    """
    专门针对多分类任务实现的加权交叉熵损失函数
    """
    def __init__(self, weight: Optional[List[float]] = None, reduction: str = 'mean'):
        super().__init__()
        self.weight = torch.tensor(weight) if weight is not None else None
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.weight is not None:
            self.weight = self.weight.to(input.device)
        return nn.functional.cross_entropy(input, target, weight=self.weight, reduction=self.reduction)
    
    
class MyBinaryCrossEntropyLoss:
    def __int__(self):
        """
        https://zhuanlan.zhihu.com/p/562641889
        专门针对二元实现的BCELoss，input是一维Tensor，label是二维，需要flatten
        :param input:
        :param target:
        :return:
        """
        pass

    def __call__(self, input, target, **kwargs):
        return BCELoss()(Sigmoid()(input.flatten()), target.to(dtype=torch.float))

class MultiClassFocalLoss(nn.Module):
    def __init__(self, alpha:List[int]|List[float], gamma=2, reduction='mean'):
        """
        :param alpha: 权重系数列表
        :param gamma: 困难样本挖掘的gamma
        :param reduction:
        """
        super(MultiClassFocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha).to(getDevice())
        self.alpha = self.alpha/self.alpha.sum() # 归一化
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        alpha = self.alpha[target]  # 为当前batch内的样本，逐个分配类别权重，shape=(bs), 一维向量
        log_softmax = torch.log_softmax(pred, dim=1) # 对模型裸输出做softmax再取log, shape=(bs, 3)
        logpt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))  # 取出每个样本在类别标签位置的log_softmax值, shape=(bs, 1)
        logpt = logpt.view(-1)  # 降维，shape=(bs)
        ce_loss = -logpt  # 对log_softmax再取负，就是交叉熵了
        pt = torch.exp(logpt)  #对log_softmax取exp，把log消了，就是每个样本在类别标签位置的softmax值了，shape=(bs)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss  # 根据公式计算focal loss，得到每个样本的loss值，shape=(bs)
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss


class MyBinaryFocalLoss:
    def __init__(self,alpha:float,gamma:float,**kwargs):
        """
        专门针对单个fea实现的FocalLoss，fea任务所用模型的输出可能含有负数
        :param input:
        :param target:
        :return:
        """
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, input, target, **kwargs):
        return sigmoid_focal_loss(input.flatten(),target.to(torch.float),self.alpha,self.gamma,'mean')
