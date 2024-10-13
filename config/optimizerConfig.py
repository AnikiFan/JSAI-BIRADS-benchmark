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

@dataclass
class AdamOptimizerConfig:
    _target_: str = "torch.optim.Adam"
    params:Any=MISSING


@dataclass
class AdamWOptimizerConfig:
    _target_: str = "torch.optim.AdamW"
    params:Any=MISSING
    lr:float=1e-4
    weight_decay:float=1e-2
    betas:Tuple[float,float]=(0.9,0.999)
    eps:float=1e-8
