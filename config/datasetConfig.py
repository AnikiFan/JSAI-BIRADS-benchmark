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
    official_train: bool = True
    BUS: bool = True
    USG: bool = True
    fea_official_train: bool = False
    _convert_: str = "all"


@dataclass
class FeaDatasetConfig:
    data_folder_path: Path = MISSING
    image_format: str = "Tensor"
    num_classes: int = 1
    official_train: bool = False
    BUS: bool = False
    USG: bool = False
    fea_official_train: bool = True
    feature: str = 'all'


@dataclass
class SingleFeaDatasetConfig:
    data_folder_path: Path = MISSING
    image_format: str = "Tensor"
    num_classes: int = 1
    official_train: bool = False
    BUS: bool = False
    USG: bool = False
    fea_official_train: bool = True


@dataclass
class BoundaryDatasetConfig(SingleFeaDatasetConfig):
    ratio: List[int] = field(default_factory=lambda: [1, 1])
    feature: str = 'boundary'


@dataclass
class CalcificationDatasetConfig(SingleFeaDatasetConfig):
    ratio: List[int] = field(default_factory=lambda: [1, 1])
    feature: str = 'calcification'


@dataclass
class DirectionDatasetConfig(SingleFeaDatasetConfig):
    ratio: List[int] = field(default_factory=lambda: [1, 1])
    feature: str = 'direction'


@dataclass
class ShapeDatasetConfig(SingleFeaDatasetConfig):
    ratio: List[int] = field(default_factory=lambda: [1, 1])
    feature: str = 'shape'


@dataclass
class ClaAugmentedDatasetConfig:
    # augmented_folder_list: List[Path] = field(
    #     default_factory=lambda: [
    #         os.path.join(os.curdir, 'data', 'breast', 'cla', 'augmented',
    #                      'VerticalFlip,ratio=(1.9,1.5,4.1,5.9,6.6,5.8)-1'),
    #         os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented', 'Mixup,ratio=(0.9,0.5,3.1,4.9,5.6,4.8)-1'),
    #         os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented', 'Mixup,ratio=(1.9,1.5,4.1,5.9,6.6,5.8)-1'),
    #     ]
    # )
    pass


@dataclass
class FeaAugmentedDatasetConfig:
    pass


@dataclass
class BoundaryAugmentedDatasetConfig:
    augmented_folder_list: List[Path] = field(default_factory=lambda: [
        # os.path.join(os.curdir,'data','breast','fea','augmented','RandomBrightnessContrast-1'),
        # os.path.join(os.curdir, 'data','breast', 'fea', 'augmented', 'GaussNoise-1'),
        # os.path.join(os.curdir, 'data','breast', 'fea', 'augmented', 'HorizontalFlip-1'),
        os.path.join(os.curdir, 'data', 'breast', 'fea', 'augmented', 'VerticalFlip-1'),
    ])


@dataclass
class CalcificationAugmentedDatasetConfig:
    pass


@dataclass
class DirectionAugmentedDatasetConfig:
    augmented_folder_list: List[Path] = field(default_factory=lambda: [
        os.path.join(os.curdir, 'data', 'breast', 'fea', 'augmented', 'RandomBrightnessContrast-1'),
        # os.path.join(os.curdir,'data','breast','fea','augmented','ElasticTransform-1'),
        os.path.join(os.curdir, 'data', 'breast', 'fea', 'augmented', 'Mixup-1'),
        os.path.join(os.curdir, 'data', 'breast', 'fea', 'augmented', 'HorizontalFlip-1'),
        os.path.join(os.curdir, 'data', 'breast', 'fea', 'augmented', 'VerticalFlip-1'),
    ])


@dataclass
class ShapeAugmentedDatasetConfig:
    augmented_folder_list: List[Path] = field(default_factory=lambda: [
        os.path.join(os.curdir, 'data', 'breast', 'fea', 'augmented', 'RandomBrightnessContrast-1'),
        # os.path.join(os.curdir,'data','breast','fea','augmented','ElasticTransform-1'),
        # os.path.join(os.curdir, 'data','breast', 'fea', 'augmented', 'GaussNoise-1'),
        os.path.join(os.curdir, 'data', 'breast', 'fea', 'augmented', 'HorizontalFlip-1'),
        os.path.join(os.curdir, 'data', 'breast', 'fea', 'augmented', 'VerticalFlip-1'),
    ])


@dataclass
class Cla4SingleFoldDatasetConfig(ClaDatasetConfig,ClaAugmentedDatasetConfig):
    _target_: str = "utils.BreastDataset.getBreastTrainValidData"
    selected_class: List[bool] = field(default_factory=lambda: [False,False,True,True,True,False])


@dataclass
class ClaSingleFoldDatasetConfig(ClaDatasetConfig, ClaAugmentedDatasetConfig):
    _target_: str = "utils.BreastDataset.getBreastTrainValidData"


@dataclass
class ClaCrossValidationDatasetConfig(ClaDatasetConfig, ClaAugmentedDatasetConfig):
    _target_: str = "utils.BreastDataset.BreastCrossValidationData"


@dataclass
class FeaSingleFoldDatasetConfig(FeaDatasetConfig, FeaAugmentedDatasetConfig):
    _target_: str = "utils.BreastDataset.getBreastTrainValidData"


@dataclass
class FeaCrossValidationDatasetConfig(FeaDatasetConfig, FeaAugmentedDatasetConfig):
    _target_: str = "utils.BreastDataset.BreastCrossValidationData"


@dataclass
class BoundarySingleFoldDatasetConfig(BoundaryDatasetConfig, BoundaryAugmentedDatasetConfig):
    _target_: str = "utils.BreastDataset.getBreastTrainValidData"


@dataclass
class BoundaryCrossValidationDatasetConfig(BoundaryDatasetConfig, BoundaryAugmentedDatasetConfig):
    _target_: str = "utils.BreastDataset.BreastCrossValidationData"


@dataclass
class CalcificationSingleFoldDatasetConfig(CalcificationDatasetConfig, CalcificationAugmentedDatasetConfig):
    _target_: str = "utils.BreastDataset.getBreastTrainValidData"


@dataclass
class CalcificationCrossValidationDatasetConfig(CalcificationDatasetConfig, CalcificationAugmentedDatasetConfig):
    _target_: str = "utils.BreastDataset.BreastCrossValidationData"


@dataclass
class DirectionSingleFoldDatasetConfig(DirectionDatasetConfig, DirectionAugmentedDatasetConfig):
    _target_: str = "utils.BreastDataset.getBreastTrainValidData"


@dataclass
class DirectionCrossValidationDatasetConfig(DirectionDatasetConfig, DirectionAugmentedDatasetConfig):
    _target_: str = "utils.BreastDataset.BreastCrossValidationData"


@dataclass
class ShapeSingleFoldDatasetConfig(ShapeDatasetConfig, ShapeAugmentedDatasetConfig):
    _target_: str = "utils.BreastDataset.getBreastTrainValidData"


@dataclass
class ShapeCrossValidationDatasetConfig(ShapeDatasetConfig, ShapeAugmentedDatasetConfig):
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
