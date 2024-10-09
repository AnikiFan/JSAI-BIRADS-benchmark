from dataclasses import dataclass
from omegaconf import MISSING
from typing import *

"""
优化器配置，用于直接实例化优化器
"""


@dataclass
class SGDOptimizerConfig:
    _target_: str = "torch.optim.SGD"
    params:Any=MISSING
