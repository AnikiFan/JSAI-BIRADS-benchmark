from dataclasses import dataclass, field
from omegaconf import MISSING
from pathlib import Path
from utils.tools import getDevice
from utils.earlyStopping import EarlyStopping
from torcheval.metrics.functional import multiclass_f1_score,multiclass_accuracy,multiclass_confusion_matrix
from torch import Tensor
"""
训练过程配置，例如损失函数，早停等参数设置
"""
@dataclass
class CrossEntropyConfig:
    _target_: str = 'torch.nn.CrossEntropyLoss'


@dataclass
class EarlyStopping:
    _target_: str = 'utils.earlyStopping.EarlyStopping'
    patience: int = 20
    min_delta: float = 0.001

@dataclass
class multiclass_accuracy:
    _target_:str = "torcheval.metrics.functional.multiclass_accuracy"
    input:Tensor = MISSING
    target:Tensor = MISSING
    average:str = 'macro'
    num_classes:int = MISSING
    pass


@dataclass
class DefaultTrainConfig:
    checkpoint_path: Path = ''
    epoch_num: int = 1000
    num_workers: int = 2
    batch_size: int = 16
    info_frequency: int = 100
    early_stopping: EarlyStopping = field(default_factory=EarlyStopping)
    loss_function: CrossEntropyConfig = field(default_factory=CrossEntropyConfig)

