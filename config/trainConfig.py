from dataclasses import dataclass, field
from omegaconf import MISSING
from pathlib import Path
from utils.tools import getDevice
from utils.earlyStopping import EarlyStopping

"""
训练过程配置，例如损失函数，早停等参数设置
"""
@dataclass
class ClaTrainConfig:
    num_classes:int = 6

@dataclass
class LossFunction:
    _target_: str = 'torch.nn.CrossEntropyLoss'


@dataclass
class EarlyStopping:
    _target_: str = 'utils.earlyStopping.EarlyStopping'
    patience: int = 20
    min_delta: float = 0.001


@dataclass
class DefaultTrainConfig(ClaTrainConfig):
    checkpoint_path: Path = ''
    epoch_num: int = 1000
    num_workers: int = 2
    batch_size: int = 16
    info_frequency: int = 100
    early_stopping: EarlyStopping = field(default_factory=EarlyStopping)
    loss_function: LossFunction = field(default_factory=LossFunction)
