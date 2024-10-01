from dataclasses import dataclass, field
from omegaconf import MISSING, OmegaConf

from hydra.core.config_store import ConfigStore
from .trainConfig import *
from .datasetConfig import *
from .modelConfig import *
from .optimizerConfig import *
from .envConfig import *
from .transformConfig import *
from .schedularConfig import *
import hydra
from typing import *


@dataclass
class Config:
    defaults: List[Any] = field(default_factory=lambda: [
        {"train": "default"},
        {"model": "linear_sanity_check"},
        {"dataset": "mnist"},
        {"optimizer": "SGD"},
        {"env": "fx"},
        {"train_transform": "default"},
        {"valid_transform": "default"},
        {"schedular": "exponential"},

        # 彩色日志插件，会导致无法自动保存日志
        # {"override hydra/job_logging": "colorlog"},
        # {"override hydra/hydra_logging": "colorlog"}
    ])
    train: Any = MISSING
    model: Any = MISSING
    dataset: Any = MISSING
    optimizer: Any = MISSING
    env: Any = MISSING
    train_transform: Any = MISSING
    valid_transform: Any = MISSING
    schedular: Any = MISSING


# TODO: 为了适配多特征识别任务，可能需要将指标评价方法也加入配置中


def init_config():
    """
    初始化配置
    :return:
    """
    # 初始化
    cs = ConfigStore.instance()

    cs.store(group='train', name="default", node=DefaultTrainConfig)

    cs.store(group='model', name="alex_net", node=AlexNetModelConfig)
    cs.store(group='model', name="google_net", node=GoogleNetModelConfig)
    cs.store(group='model', name="NiN", node=NiNModelConfig)
    cs.store(group='model', name="VGG", node=VGGModelConfig)
    cs.store(group='model', name="linear_sanity_check", node=LinearSanityCheckerModelConfig)
    cs.store(group='model', name="conv_sanity_check", node=ConvSanityCheckerModelConfig)
    cs.store(group='model', name="default", node=DefaultModelConfig)

    cs.store(group='dataset', name="fashion_mnist", node=FashionMNISTDatasetConfig)
    cs.store(group='dataset', name="mnist", node=MNISTDatasetConfig)
    cs.store(group='dataset', name="cifar10", node=CIFAR10DatasetConfig)
    cs.store(group='dataset', name="single", node=SingleFoldDatasetConfig)
    cs.store(group='dataset', name="multiple", node=CrossValidationDatasetConfig)

    cs.store(group='optimizer', name="SGD", node=SGDOptimizerConfig)

    cs.store(group='env', name="fx", node=FXEnvConfig)
    cs.store(group='env', name="zhy", node=ZHYEnvConfig)
    cs.store(group='env', name="yzl", node=YZLEnvConfig)

    cs.store(group='train_transform', name="default", node=DefaultTrainTransformConfig)

    cs.store(group='valid_transform', name="default", node=DefaultValidTransformConfig)

    cs.store(group='schedular', name="exponential", node=ExponentialLRConfig)
    cs.store(group='schedular', name="dummy", node=DummySchedularConfig)

    # 初始化
    cs.store(name="config", node=Config)
