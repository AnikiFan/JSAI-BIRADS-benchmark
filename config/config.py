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
    {"train": "sanity_check"},
    {"model": "sanity_check"},
    {"dataset": "sanity_check"},
    {"optimizer": "default"},
    {"env": "fx"},
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
    # 初始化
    cs = ConfigStore.instance()

    cs.store(group='train', name="default", node=DefaultTrainConfig)
    cs.store(group='train', name="sanity_check", node=FashionMNISTTrainConfig)

    cs.store(group='model', name="default", node=DefaultModelConfig)
    cs.store(group='model', name="sanity_check", node=AlexNetModelConfig)

    cs.store(group='dataset', name="single", node=SingleFoldDatasetConfig)
    cs.store(group='dataset', name="multiple", node=CrossValidationDatasetConfig)
    cs.store(group='dataset', name="sanity_check", node=FashionMNISTDatasetConfig)

    cs.store(group='optimizer', name="default", node=DefaultOptimizerConfig)

    cs.store(group='env', name="fx", node=FXEnvConfig)
    cs.store(group='env', name="zhy", node=ZHYEnvConfig)
    cs.store(group='env', name="yzl", node=YZLEnvConfig)

    cs.store(group='train_transform', name="default", node=DefaultTrainTransformConfig)

    cs.store(group='valid_transform', name="default", node=DefaultValidTransformConfig)

    # 初始化
    cs.store(name="config", node=Config)
