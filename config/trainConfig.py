from dataclasses import dataclass, field
from omegaconf import MISSING
from pathlib import Path
from utils.tools import getDevice
from utils.earlyStopping import EarlyStopping

@dataclass
class LossFunction:
    _target_: str='torch.nn.CrossEntropyLoss'


@dataclass
class Debug:
    num_smaples_to_show: int = 4


@dataclass
class EarlyStopping:
    _target_: str='utils.earlyStopping.EarlyStopping'
    patience: int = 20
    min_delta: float = 0.001


@dataclass
class DatasetRoot:
    train_dir: Path = Path(
        "/Users/huiyangzheng/Desktop/Project/Competition/GCAIAEC2024/AIC/TDS-Net/data/乳腺分类训练数据集/train_split/train_split_train")
    test_dir: Path = Path(
        "/Users/huiyangzheng/Desktop/Project/Competition/GCAIAEC2024/AIC/TDS-Net/data/乳腺分类训练数据集/train_split/train_split_test")
    split_ratio: float = 0.9



@dataclass
class DefaultTrainConfig:
    checkpoint_path: Path = ''
    epoch_num: int = 1000
    num_workers: int = 2
    batch_size: int = 16
    in_channels: int = 3
    num_classes:int = 6
    info_show_frequency: int = 100
    dataset_root: Path = "/Users/huiyangzheng/Desktop/Project/Competition/GCAIAEC2024/AIC/TDS-Net/data/乳腺分类训练数据集/train_split"

    debug: Debug = field(default_factory=Debug)

    early_stopping: EarlyStopping = field(default_factory=EarlyStopping)

    dataset_root: DatasetRoot = field(default_factory=DatasetRoot)

    loss_function :LossFunction = field(default_factory=LossFunction)

    device: str = getDevice()
    pin_memory: bool = getDevice() == "cuda"

    resume:bool=False