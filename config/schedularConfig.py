from dataclasses import dataclass, field
from omegaconf import MISSING
from pathlib import Path
from utils.tools import getDevice
from torch.optim import SGD
from typing import *
from torch.optim.lr_scheduler import ExponentialLR
from utils.schedular import DummyScheduler

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
    _target_: str = "utils.schedular.DummyScheduler"
