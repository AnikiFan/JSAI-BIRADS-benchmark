from dataclasses import dataclass
from omegaconf import MISSING
from typing import *

"""
用于实例化lr_schedular的配置
"""


@dataclass
class ExponentialLRConfig:
    _target_: str = "torch.optim.lr_scheduler.ExponentialLR"
    optimizer: Any = MISSING
    gamma: float = 0.9


@dataclass
class DummySchedularConfig:
    """
    不对lr有任何调整的schedular
    """
    _target_: str = "utils.schedular.DummyScheduler"
