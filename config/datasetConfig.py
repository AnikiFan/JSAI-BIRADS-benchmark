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
数据集配置，用于直接实例化数据集，所以不能有数据集所需的参数以外的配置项
数据集对象应该是一个迭代器，每次迭代返回train_dataset和valid_dataset
"""


@dataclass
class ClaDatasetConfig:
    data_folder_path: Path = MISSING
    image_format: str = "Tensor"
    num_classes: int = 6

@dataclass
class ClaAugmentedDatasetConfig:
    augmented_folder_list: List[Path] = field(
        default_factory=lambda: [
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented', 'CLAHE,ratio=(1,1,1,1,1,1)-1'),
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented', 'ElasticTransform,ratio=(1,1,1,1,1,1)-1'),
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented', 'GaussianBlur,ratio=(1,1,1,1,1,1)-1'),
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented', 'GaussNoise,ratio=(1,1,1,1,1,1)-1'),
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented', 'HorizontalFlip,ratio=(1,1,1,1,1,1)-1'),
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented', 'Mixup,ratio=(1,1,1,1,1,1)-1'),
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented', 'Mixup,ratio=(1,1,1,1,1,1)-2'),
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented', 'RandomBrightnessContrast,ratio=(1,1,1,1,1,1)-1'),
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented', 'Rotate,ratio=(1,1,1,1,1,1)-1'),
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented', 'ShiftScaleRotate,ratio=(1,1,1,1,1,1)-1'),
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented', 'VerticalFlip,ratio=(1,1,1,1,1,1)-1')
        ]
    )

@dataclass
class SingleFoldDatasetConfig(ClaDatasetConfig,ClaAugmentedDatasetConfig):
    _target_: str = "utils.BreastDataset.getBreastTrainValidData"
    _convert_: str = "all"
    



@dataclass
class CrossValidationDatasetConfig(ClaDatasetConfig):
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
