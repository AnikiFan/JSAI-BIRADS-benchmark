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
    """
    初始化配置
    :return:
    """
    cs = ConfigStore.instance()
    cs.store(name="config", node=Config)


    # 训练配置，如epoch数量，早停机制等
    cs.store(group="train", name="default", node=DefaultTrainConfig)
    # 模型配置
    cs.store(group="model", name="default", node=DefaultModelConfig)
    # 数据集配置
    cs.store(group="dataset", name="default", node=DefaultDatasetConfig)
    # 优化器配置
    cs.store(group="optimizer", name="default", node=DefaultOptimizerConfig)
    # 环境配置
    cs.store(group="env", name="default", node=DefaultEnvConfig)
    # 训练集图像所用变换配置
    cs.store(group="train_transform", name="default", node=DefaultTrainTransformConfig)
    # 验证集图像所用变换配置
    cs.store(group="valid_transform", name="default", node=DefaultValidTransformConfig)


@hydra.main(version_base=None, config_name="config")
def test_config(cfg: Config):
    """
    测试配置
    :param cfg:
    :return:
    """
    print("配置清单：")
    print(OmegaConf.to_yaml(cfg))
    print("输出保存路径：")
    print(cfg.train.dataset_root.train_dir)
    return