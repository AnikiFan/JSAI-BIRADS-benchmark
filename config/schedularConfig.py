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


@dataclass
class StepLRConfig:
    """
    每隔特定的epoch数，按一定的gamma衰减学习率
    """
    _target_: str = "torch.optim.lr_scheduler.StepLR"
    optimizer: Any = MISSING
    step_size: int = 30
    gamma: float = 0.1


@dataclass
class MultiStepLRConfig:
    """
    在预定义的epoch列表中，每到一个epoch就按gamma衰减学习率
    """
    _target_: str = "torch.optim.lr_scheduler.MultiStepLR"
    optimizer: Any = MISSING
    milestones: List[int] = MISSING
    gamma: float = 0.1


@dataclass
class CosineAnnealingLRConfig:
    """
    使用余弦退火方法调整学习率
    """
    _target_: str = "torch.optim.lr_scheduler.CosineAnnealingLR"
    optimizer: Any = MISSING
    T_max: int = 50
    eta_min: float = 0.0001


@dataclass
class ReduceLROnPlateauConfig:
    """
    当指标停止提升时，降低学习率
    """
    _target_: str = "torch.optim.lr_scheduler.ReduceLROnPlateau"
    optimizer: Any = MISSING
    mode: str = "min"
    factor: float = 0.1
    patience: int = 10
    threshold: float = 1e-4
