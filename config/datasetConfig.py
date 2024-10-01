from dataclasses import dataclass, field
from omegaconf import MISSING
from pathlib import Path
from utils.BreastDataset import getBreastTrainValidData, BreastCrossValidationData
import os
from data.FashionMNIST.MyFashionMNIST import MyFashionMNIST
from data.MNIST.MyMNIST import MyMNIST
from data.CIFAR10.MyCIFAR10 import MyCIFAR10

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
class SingleFoldDatasetConfig(ClaDatasetConfig):
    _target_: str = "utils.BreastDataset.getBreastTrainValidData"


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

