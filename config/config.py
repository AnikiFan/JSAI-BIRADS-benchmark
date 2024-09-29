from dataclasses import dataclass, field
from omegaconf import MISSING, OmegaConf

from hydra.core.config_store import ConfigStore
from .trainConfig import *
from .datasetConfig import *
from .modelConfig import *
from .optimizerConfig import *
from .envConfig import *
from .transformConfig import *
import hydra
from typing import *

defaults = [
    {"train": "default"},
    {"model": "default"},
    {"dataset": "default"},
    {"optimizer": "default"},
    {"env": "default"},
    {"train_transform": "default"},
    {"valid_transform": "default"}
]


@dataclass
class Config:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    train: Any = MISSING
    model: Any = MISSING
    dataset: Any = MISSING
    optimizer: Any = MISSING
    env: Any = MISSING
    train_transform: Any = MISSING
    valid_transform: Any = MISSING


def init_config():
    """
    初始化配置
    :return:
    """
    cs = ConfigStore.instance()
    cs.store(group='train', name="default", node=DefaultTrainConfig)
    cs.store(group='model', name="default", node=DefaultModelConfig)
    cs.store(group='dataset', name="default", node=DefaultDatasetConfig)
    cs.store(group='optimizer', name="default", node=DefaultOptimizerConfig)
    cs.store(group='env', name="default", node=DefaultEnvConfig)
    cs.store(group='train_transform', name="default", node=DefaultTrainTransformConfig)
    cs.store(group='valid_transform', name="default", node=DefaultValidTransformConfig)
    cs.store(name="config", node=Config)
