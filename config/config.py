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


@dataclass
class Config:
    # We will populate db using composition.
    train: DefaultTrainConfig = field(default_factory=DefaultTrainConfig)
    model: DefaultModelConfig = field(default_factory=DefaultModelConfig)
    dataset: DefaultDatasetConfig = field(default_factory=DefaultDatasetConfig)
    optimizer: DefaultOptimizerConfig = field(default_factory=DefaultOptimizerConfig)
    env: DefaultEnvConfig = field(default_factory=DefaultEnvConfig)
    train_transform: DefaultTrainTransformConfig = field(default_factory=DefaultTrainTransformConfig)
    valid_transform: DefaultValidTransformConfig = field(default_factory=DefaultValidTransformConfig)


def init_config():
    cs = ConfigStore.instance()
    cs.store(name="config", node=Config)
    cs.store(group="train", name="default", node=DefaultTrainConfig)
    cs.store(group="model", name="default", node=DefaultModelConfig)
    cs.store(group="dataset", name="default", node=DefaultDatasetConfig)
    cs.store(group="optimizer", name="default", node=DefaultOptimizerConfig)
    cs.store(group="env", name="default", node=DefaultEnvConfig)
    cs.store(group="train_transform", name="default", node=DefaultTrainTransformConfig)
    cs.store(group="valid_transform", name="default", node=DefaultValidTransformConfig)


@hydra.main(version_base=None, config_name="config")
def test_config(cfg: Config):
    print(OmegaConf.to_yaml(cfg))
    print(cfg.train.dataset_root.train_dir)
    return