import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


# 交叉
def CE_Loss(inputs, target, cls_weights, num_classes=21):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    CE_loss  = nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes)(temp_inputs, temp_target)
    return CE_loss

# 焦点损失
def Focal_Loss(inputs, target, cls_weights, num_classes=21, alpha=0.5, gamma=2):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    logpt  = -nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes, reduction='none')(temp_inputs, temp_target)
    pt = torch.exp(logpt)
    if alpha is not None:
        logpt *= alpha
    loss = -((1 - pt) ** gamma) * logpt 
    loss = loss.mean()
    return loss

# Dice损失
def Dice_loss(inputs, target, beta=1, smooth = 1e-5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
        
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
    temp_target = target.view(n, -1, ct)

    #--------------------------------------------#
    #   计算dice loss
    #--------------------------------------------#
    tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0,1])
    fp = torch.sum(temp_inputs                       , axis=[0,1]) - tp
    fn = torch.sum(temp_target[...,:-1]              , axis=[0,1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    dice_loss = 1 - torch.mean(score)
    return dice_loss


# 初始化网络权重
def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)


# 
"""
get_lr_scheduler 函数使用说明

该函数用于生成一个学习率调度器函数，该调度器函数可以根据当前的迭代次数动态调整学习率。该函数支持两种学习率调度策略：余弦退火调度和阶梯调度。

参数:
- lr_decay_type (str): 学习率衰减类型，可选值为 "cos" 或 "step"。
  - "cos": 使用余弦退火调度策略。
  - "step": 使用阶梯调度策略。
- lr (float): 初始学习率。
- min_lr (float): 最小学习率。
- total_iters (int): 总迭代次数。
- warmup_iters_ratio (float, optional): 预热阶段的迭代次数比例，默认为 0.05。
- warmup_lr_ratio (float, optional): 预热阶段的学习率比例，默认为 0.1。
- no_aug_iter_ratio (float, optional): 无数据增强阶段的迭代次数比例，默认为 0.05。
- step_num (int, optional): 阶梯调度的步数，默认为 10。

返回值:
- func (function): 返回一个部分应用函数，该函数接受当前迭代次数 iters 作为参数，并返回相应的学习率。

使用示例:
以下是如何使用 get_lr_scheduler 函数生成学习率调度器并在训练过程中调整学习率的示例：

import torch.optim as optim

# 假设我们有一个优化器
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 生成学习率调度器
lr_scheduler_func = get_lr_scheduler(
    lr_decay_type="cos",  # 使用余弦退火调度
    lr=0.1,               # 初始学习率
    min_lr=0.001,         # 最小学习率
    total_iters=1000,     # 总迭代次数
    warmup_iters_ratio=0.1,  # 预热阶段的迭代次数比例
    warmup_lr_ratio=0.1,     # 预热阶段的学习率比例
    no_aug_iter_ratio=0.05,  # 无数据增强阶段的迭代次数比例
    step_num=10              # 阶梯调度的步数
)

# 在训练过程中调整学习率
for epoch in range(total_epochs):
    for i, data in enumerate(train_loader):
        # 获取当前迭代次数
        current_iter = epoch * len(train_loader) + i
        
        # 计算当前学习率
        current_lr = lr_scheduler_func(current_iter)
        
        # 设置优化器的学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        # 训练步骤
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
"""
def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
