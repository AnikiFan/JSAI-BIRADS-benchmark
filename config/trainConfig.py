from dataclasses import dataclass, field
from omegaconf import MISSING
from pathlib import Path
from typing import *

"""
训练过程配置，例如损失函数，早停等参数设置
"""


@dataclass
class CrossEntropyLossConfig:
    _target_: str = 'torch.nn.CrossEntropyLoss'

@dataclass
class BCELossConfig:
    _target_: str = 'utils.loss_function.MyBCELoss'


@dataclass
class EarlyStopping:
    _target_: str = 'utils.earlyStopping.EarlyStopping'
    patience: int = 10
    min_delta: float = 0.001
    min_train_loss: float = 0.3

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
    _target_: str = "utils.metrics.my_multilabel_accuracy"
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
class ScoreOrientedConfig:
    _target_:str="utils.ChooseStrategy.score_oriented"
    loss:Any=MISSING
    accuracy:Any=MISSING
    f1_score:Any=MISSING

@dataclass
class MultiClassTrainConfig:
    accuracy:MultiClassAccuracy = field(default_factory=MultiClassAccuracy)
    f1_score:MultiClassF1Score = field(default_factory=MultiClassF1Score)
    confusion_matrix:MultiClassConfusionMatrix = field(default_factory=MultiClassConfusionMatrix)
    choose_strategy:ScoreOrientedConfig = field(default_factory=ScoreOrientedConfig)


@dataclass
class MultiLabelTrainConfig:
    accuracy: MultiLabelAccuracy = field(default_factory=MultiLabelAccuracy)
    f1_score: MultiLabelF1Score = field(default_factory=MultiLabelF1Score)
    confusion_matrix: MultiLabelConfusionMatrix = field(default_factory=MultiLabelConfusionMatrix)
    choose_strategy:ScoreOrientedConfig = field(default_factory=ScoreOrientedConfig)

@dataclass
class DefaultTrainConfig:
    epoch_num: int = 1000
    num_workers: int = 4
    batch_size: int = 16
    info_frequency: int = 100
    early_stopping: EarlyStopping = field(default_factory=EarlyStopping)


@dataclass
class ClaTrainConfig(MultiClassTrainConfig, DefaultTrainConfig):
    loss_function: CrossEntropyLossConfig = field(default_factory=CrossEntropyLossConfig)


@dataclass
class FeaTrainConfig(MultiLabelTrainConfig, DefaultTrainConfig):
    loss_function: BCELossConfig = field(default_factory=BCELossConfig)


@dataclass
class RemoteTrainConfig(ClaTrainConfig):
    checkpoint_path: Path = MISSING
    epoch_num: int = 1000
    num_workers: int = 10
    batch_size: int = 16
    info_frequency: int = 100
    early_stopping: EarlyStopping = field(default_factory=EarlyStopping)
    loss_function: CrossEntropyLossConfig = field(default_factory=CrossEntropyLossConfig)

