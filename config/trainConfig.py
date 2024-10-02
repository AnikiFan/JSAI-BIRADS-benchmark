from dataclasses import dataclass, field
from omegaconf import MISSING
from pathlib import Path
from typing import *
from utils.earlyStopping import EarlyStopping

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
class MultiClassAccuracy:
    _target_: str = "torcheval.metrics.functional.multiclass_accuracy"
    input: Any = MISSING
    target: Any = MISSING
    average: str = 'macro'
    num_classes: int = MISSING


@dataclass
class MultiClassF1Score:
    _target_: str = "torcheval.metrics.functional.multiclass_f1_score"
    input: Any = MISSING
    target: Any = MISSING
    average: str = 'macro'
    num_classes: int = MISSING


@dataclass
class MultiClassConfusionMatrix:
    _target_: str = "torcheval.metrics.functional.multiclass_confusion_matrix"
    input: Any = MISSING
    target: Any = MISSING
    num_classes: int = MISSING


@dataclass
class MultiLabelAccuracy:
    _target_: str = "torcheval.metrics.functional.multilabel_accuracy"
    input: Any = MISSING
    target: Any = MISSING
    criteria: str = 'hamming'


@dataclass
class MultiLabelF1Score:
    _target_: str = "utils.metrics.multilabel_f1_score"
    input: Any = MISSING
    target: Any = MISSING


@dataclass
class MultiLabelConfusionMatrix:
    _target_: str = "utils.metrics.multilabel_confusion_matrix"
    input: Any = MISSING
    target: Any = MISSING


@dataclass
class MultiClassTrainConfig:
    accuracy:MultiClassAccuracy = field(default_factory=MultiClassAccuracy)
    f1_score:MultiClassF1Score = field(default_factory=MultiClassF1Score)
    confusion_matrix:MultiClassConfusionMatrix = field(default_factory=MultiClassConfusionMatrix)


@dataclass
class MultiLabelTrainConfig:
    accuracy: MultiLabelAccuracy = field(default_factory=MultiLabelAccuracy)
    f1_score: MultiLabelF1Score = field(default_factory=MultiLabelF1Score)
    confusion_matrix: MultiLabelConfusionMatrix = field(default_factory=MultiLabelConfusionMatrix)

@dataclass
class DefaultTrainConfig(MultiClassTrainConfig):
    checkpoint_path: Path = MISSING
    epoch_num: int = 1000
    num_workers: int = 2
    batch_size: int = 16
    info_frequency: int = 100
    early_stopping: EarlyStopping = field(default_factory=EarlyStopping)
    loss_function: CrossEntropyConfig = field(default_factory=CrossEntropyConfig)
