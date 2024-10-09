from dataclasses import dataclass, field
from omegaconf import MISSING
from pathlib import Path
from utils.BreastDataset import getBreastTrainValidData, BreastCrossValidationData
import os
from data.FashionMNIST.MyFashionMNIST import MyFashionMNIST
from data.MNIST.MyMNIST import MyMNIST
from data.CIFAR10.MyCIFAR10 import MyCIFAR10
from typing import *

"""
数据集配置，用于直接实例化数据集
数据集对象应该是一个迭代器，每次迭代返回train_dataset和valid_dataset
"""


@dataclass
class ClaDatasetConfig:
    data_folder_path: Path = MISSING
    image_format: str = "Tensor"
    num_classes: int = 6
    official_train:bool=True
    BUS:bool=True
    USG:bool=True
    fea_official_train:bool=False

@dataclass
class FeaDatasetConfig:
    data_folder_path: Path = MISSING
    image_format: str = "Tensor"
    num_classes: int = 2
    official_train:bool=False
    BUS:bool=False
    USG:bool=False
    fea_official_train:bool=True


@dataclass
class ClaAugmentedDatasetConfig:
    augmented_folder_list: List[Path] = field(
        default_factory=lambda:[
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented', 'Mixup,ratio=(2,1,3,4,5,6)'),
            os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented', 'VerticalFlip,ratio=(2,1,3,4,5,6)')
        ])

@dataclass
class FeaAugmentedDatasetConfig:
    pass

@dataclass
class ClaSingleFoldDatasetConfig(ClaDatasetConfig,ClaAugmentedDatasetConfig):
    _target_: str = "utils.BreastDataset.getBreastTrainValidData"


@dataclass
class ClaCrossValidationDatasetConfig(ClaDatasetConfig,ClaAugmentedDatasetConfig):
    _target_: str = "utils.BreastDataset.BreastCrossValidationData"

@dataclass
class FeaSingleFoldDatasetConfig(FeaDatasetConfig, FeaAugmentedDatasetConfig):
    _target_: str = "utils.BreastDataset.getBreastTrainValidData"

@dataclass
class FeaCrossValidationDatasetConfig(FeaDatasetConfig,FeaAugmentedDatasetConfig):
    _target_: str = "utils.BreastDataset.BreastCrossValidationData"


@dataclass
class FashionMNISTDatasetConfig:
    _target_: str = "data.FashionMNIST.MyFashionMNIST.MyFashionMNIST"
    num_classes: int = 10


@dataclass
class MNISTDatasetConfig:
    _target_: str = "data.MNIST.MyMNIST.MyMNIST"
    num_classes: int = 10


@dataclass
class CIFAR10DatasetConfig:
    _target_: str = "data.CIFAR10.MyCIFAR10.MyCIFAR10"
    num_classes: int = 10
