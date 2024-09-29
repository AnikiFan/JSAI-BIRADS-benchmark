from dataclasses import dataclass, field
from omegaconf import MISSING
from pathlib import Path
from utils.tools import getDevice
from torch.optim import SGD
from typing import *
"""
优化器配置，用于直接实例化优化器
模型参数利用_partial_在应用时传入
"""

@dataclass
class DefaultOptimizerConfig:
    """
    这里_partial_=True是为了推迟实例化，当在应用中显式地用instantiate函数时再实例化，这是为了
    传入params，这里设置的【】只是为了避免报错
    """
    _partial_:bool=True
    _target_:str="torch.optim.SGD"
    lr:float=0.1
    params:Any=field(default_factory=lambda :[])

