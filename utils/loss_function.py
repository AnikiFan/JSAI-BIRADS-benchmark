from torch.nn import BCELoss, Sigmoid
from torch import Tensor


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
