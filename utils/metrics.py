from torch import Tensor,argmax
import torch
from torcheval.metrics.functional import multilabel_accuracy
def my_multilabel_accuracy(input:Tensor,target:Tensor,**kwargs):
    return multilabel_accuracy(input,target,criteria='hamming',threshold=0)
    
def multilabel_f1_score(input:Tensor,target:Tensor,**kwargs)-> Tensor:
    """
    用于多个二元标签样本的f1score，只是将各个标签分开计算二元标签f1score，然后求平均
    :param input: 预测，可以是(num_sample,num_label,num_class=2)，也可以是(num_sample,num_label)，这里默认都是二元标签
    :param target: ground_truth，形状为(num_sample,num_label)
    :return:
    """
    assert input.shape == target.shape,"input and target shapes must match"
    input = torch.where(input<0,0,1)
    true_positive = (input*target).sum(dim=0)
    false_positive = (input*(1-target)).sum(dim=0)
    false_negative = ((1-input)*target).sum(dim=0)
    f1 = 2*true_positive/(2*true_positive+false_positive+false_negative)
    return f1.mean()

def multilabel_confusion_matrix(input:Tensor,target:Tensor,**kwargs)->Tensor:
    """
    用于多个二元标签样本的ocnfusion_matrix，只是将各个标签分开计算confusion_matrix，然后拼接起来
    :param input: 预测，可以是(num_sample,num_label,num_class=2)，也可以是(num_sample,num_label)，这里默认都是二元标签
    :param target: ground_truth，形状为(num_sample,num_label)
    :return: [TP , FN]
             [FP , TN]
    """
    assert input.shape == target.shape,"input and target shapes must match"
    input = torch.where(input<0,0,1)
    num_labels = input.shape[1]
    confusion_matrices = torch.zeros((num_labels, 2, 2), dtype=torch.int64)
    # 遍历每个标签，计算其混淆矩阵
    for i in range(num_labels):
        # 计算 True Positives, True Negatives, False Positives, False Negatives
        tp = ((input[:, i] == 1) & (target[:, i] == 1)).sum().item()
        tn = ((input[:, i] == 0) & (target[:, i] == 0)).sum().item()
        fp = ((input[:, i] == 1) & (target[:, i] == 0)).sum().item()
        fn = ((input[:, i] == 0) & (target[:, i] == 1)).sum().item()

        # 填充混淆矩阵
        confusion_matrices[i, 1, 1] = tn  # True Negatives
        confusion_matrices[i, 1, 0] = fp  # False Positives
        confusion_matrices[i, 0, 1] = fn  # False Negatives
        confusion_matrices[i, 0, 0] = tp  # True Positives

    return confusion_matrices