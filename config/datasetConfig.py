from dataclasses import dataclass, field
from omegaconf import MISSING
from pathlib import Path
from utils.ClaDataset import getClaTrainValidData, ClaCrossValidationData
import os
from data.FashionMNIST.MyFashionMNIST import MyFashionMNIST

"""
数据集配置，用于直接实例化数据集，所以不能有数据集所需的参数以外的配置项
数据集对象应该是一个迭代器，每次迭代返回train_dataset和valid_dataset
"""


@dataclass
class DatasetConfig:
    data_folder_path: Path = Path()
    image_format: str = "Tensor"


@dataclass
class SingleFoldDatasetConfig(DatasetConfig):
    _target_: str = "utils.ClaDataset.getClaTrainValidData"


@dataclass
class CrossValidationDatasetConfig(DatasetConfig):
    _target_: str = "utils.ClaDataset.ClaCrossValidationData"

@dataclass
class FashionMNISTDatasetConfig:
    _target_: str = "data.FashionMNIST.MyFashionMNIST.MyFashionMNIST"



