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
    trainROI:bool=True
    
    fea_official_train: bool = False
    ratio: List[int] = field(default_factory=lambda :[1,1,1,1,1,1])
    _convert_:str = "all"


@dataclass
class FeaDatasetConfig:
    data_folder_path: Path = MISSING
    image_format: str = "Tensor"
    num_classes: int = 1
    official_train: bool = False
    BUS: bool = False
    USG: bool = False
    trainROI:bool=False
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


class BoundaryDatasetConfig(SingleFeaDatasetConfig):
    # BUG： 不知道为什么在这里写的配置无法被继承
    feature: str = 'boundary'


class CalcificationDatasetConfig(SingleFeaDatasetConfig):
    feature: str = 'calcification'


class DirectionDatasetConfig(SingleFeaDatasetConfig):
    feature: str = 'direction'


class ShapeDatasetConfig(SingleFeaDatasetConfig):
    feature: str = 'shape'


@dataclass
class ClaAugmentedDatasetConfig:
    augmented_folder_list: List[Path] = field(
        default_factory=lambda: [
            # ratio=(0.9,1.2,2.5,3.6,3.7,4.3)
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented', 'CLAHE,ratio=(0.9,1.2,2.5,3.6,3.7,4.3)-1'),
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented', 'ElasticTransform,ratio=(0.9,1.2,2.5,3.6,3.7,4.3)-1'),
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented', 'GaussianBlur,ratio=(0.9,1.2,2.5,3.6,3.7,4.3)-1'),
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented', 'GaussNoise,ratio=(0.9,1.2,2.5,3.6,3.7,4.3)-1'),
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented', 'HorizontalFlip,ratio=(0.9,1.2,2.5,3.6,3.7,4.3)-1'),
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented', 'RandomBrightnessContrast,ratio=(0.9,1.2,2.5,3.6,3.7,4.3)-1'),
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented', 'Rotate,ratio=(0.9,1.2,2.5,3.6,3.7,4.3)-1'),
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented', 'ShiftScaleRotate,ratio=(0.9,1.2,2.5,3.6,3.7,4.3)-1'),
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented', 'VerticalFlip,ratio=(0.9,1.2,2.5,3.6,3.7,4.3)-1'),
            
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented', 'RandomBrightnessContrast,ratio=(0.9,1.2,2.5,3.6,3.7,4.3)-1'),
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented', 'Mixup,ratio=(0.9,1.2,2.5,3.6,3.7,4.3)-1'),
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented', 'Mixup,ratio=(0.9,1.2,2.5,3.6,3.7,4.3)-2'),
            # ratio=(2,1,8,7,7,6)
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented', 'GaussianBlur,ratio=(2,1,8,7,7,6)-1'),
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented', 'GaussNoise,ratio=(2,1,8,7,7,6)-1'),
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented', 'HorizontalFlip,ratio=(0.9,1.2,2.5,3.6,3.7,4.3)-1'),
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented', 'Mixup,ratio=(2,1,8,7,7,6)-1'),
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented', 'Mixup,ratio=(2,1,8,7,7,6)-2'),
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented', 'RandomBrightnessContrast,ratio=(0.9,1.2,2.5,3.6,3.7,4.3)-1'),
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented', 'Rotate,ratio=(0.9,1.2,2.5,3.6,3.7,4.3)-1'),
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented', 'ShiftScaleRotate,ratio=(0.9,1.2,2.5,3.6,3.7,4.3)-1'),
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented', 'VerticalFlip,ratio=(0.9,1.2,2.5,3.6,3.7,4.3)-1'),
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented', 'VerticalFlip,ratio=(2,1,8,7,7,6)-1'),
            
            #balance 
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented','HorizontalFlip,ratio=(0.8,0.4,2.7,4.5,5.5,4.6)-1'),
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented','Rotate,ratio=(0.4,0.1,2.0,3.4,4.0,3.4)-1'),
            # balance 
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented','Mixup,ratio=(0.9,0.5,3.1,4.9,5.6,4.8)-1'),
            
            # balance
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented','Mixup,ratio=(1.9,1.2,5.1,7.8,9.0,7.7)-2'),
            
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'au
            # gmented','Mixup,ratio=(0.4,0.1,2.0,3.4,4.0,3.4)-1'),
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented','Mixup,ratio=(0.4,0.1,2.0,3.4,4.0,3.4)-2'),
            
            # 200
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented','ElasticTransform,ratio=(0.2,0.1,0.4,0.6,0.7,0.6)-1'),
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented','GaussNoise,ratio=(0.2,0.1,0.4,0.6,0.7,0.6)-1'),
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented','HorizontalFlip,ratio=(0.2,0.1,0.4,0.6,0.7,0.6)-1'),
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented','RandomBrightnessContrast,ratio=(0.2,0.1,0.4,0.6,0.7,0.6)-1'),
            # os.path.join(os.curdir,"data", 'breast', 'cla', 'augmented','RandomGamma,ratio=(0.2,0.1,0.4,0.6,0.7,0.6)-1'),
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented','Rotate,ratio=(0.2,0.1,0.4,0.6,0.7,0.6)-1'),
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented','ShiftScaleRotate,ratio=(0.2,0.1,0.4,0.6,0.7,0.6)-1'),
            # 200 
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented','ElasticTransform,ratio=(0.2,0.2,0.5,0.7,0.9,0.7)-1'),
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented','GaussNoise,ratio=(0.2,0.2,0.5,0.7,0.9,0.7)-1'),
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented','HorizontalFlip,ratio=(0.2,0.2,0.5,0.7,0.9,0.7)-1'),
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented','RandomBrightnessContrast,ratio=(0.2,0.2,0.5,0.7,0.9,0.7)-1'),
            # os.path.join(os.curdir,"data", 'breast', 'cla', 'augmented','RandomGamma,ratio=(0.2,0.2,0.5,0.7,0.9,0.7)-1'),
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented','Rotate,ratio=(0.2,0.2,0.5,0.7,0.9,0.7)-1'),
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented','ShiftScaleRotate,ratio=(0.2,0.2,0.5,0.7,0.9,0.7)-1'),
            
            #500
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented','Rotate,ratio=(0.5,0.4,1.0,1.5,1.7,1.5)-1'),
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented','HorizontalFlip,ratio=(0.5,0.4,1.0,1.5,1.7,1.5)-1'),
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented','Mixup,ratio=(0.5,0.4,1.0,1.5,1.7,1.5)-1'),
            
            # local 
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented','Mixup,ratio=(0.9,0.5,3.1,4.9,5.6,4.8)-1'),
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented','Mixup,ratio=(0.9,0.5,3.1,4.9,5.6,4.8)-2'),
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented','Mixup,ratio=(1.9,1.5,4.1,5.9,6.6,5.8)-1'),
            
            # os.path.join(os.curdir, "data", 'breast', 
            # 'cla', 'trainROI'),
            os.path.join(os.curdir, "data", 'breast', 'cla', 'trainROI_1.5'),
            os.path.join(os.curdir, "data", 'breast', 'cla', 'trainROI_2'),
            os.path.join(os.curdir, "data", 'breast', 'cla', 'trainROI_2.5'),
            os.path.join(os.curdir, "data", 'breast', 'cla', 'trainROI_3'),
            os.path.join(os.curdir, "data", 'breast', 'cla', 'trainROI_3.5'),
            os.path.join(os.curdir, "data", 'breast', 'cla', 'trainROI_4'),
            os.path.join(os.curdir, "data", 'breast', 'cla', 'trainROI_4.5'),
            os.path.join(os.curdir, "data", 'breast', 'cla', 'trainROI_5'),
            os.path.join(os.curdir, "data", 'breast', 'cla', 'trainROI_5.5'),
            os.path.join(os.curdir, "data", 'breast', 'cla', 'trainROI_6'),
            os.path.join(os.curdir, "data", 'breast', 'cla', 'trainROI_6.5'),
            os.path.join(os.curdir, "data", 'breast', 'cla', 'trainROI_7'),
            os.path.join(os.curdir, "data", 'breast', 'cla', 'trainROI_7.5'),
            os.path.join(os.curdir, "data", 'breast', 'cla', 'trainROI_8'),
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'trainROI_MyFill2_256'),

            
            
            #!augmented_ROI 
            
            # balance
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented_ROI','Mixup,ratio=(0.4,0.1,2.0,3.4,4.0,3.4)-2'),
            os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented_ROI','Rotate,ratio=(1.4,0.3,1.3,2.6,4.0,6.8)-1'),
            
            # 500
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented_ROI','ElasticTransform,ratio=(1.4,0.3,1.3,2.6,4.0,6.8)-1'),
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented_ROI','GaussNoise,ratio=(1.4,0.3,1.3,2.6,4.0,6.8)-1'),
            os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented_ROI','HorizontalFlip,ratio=(1.4,0.3,1.3,2.6,4.0,6.8)-1'),
            os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented_ROI','MyFill2_albumentations,ratio=(9.5,0.4,9.2,18.2,28.9,50.2)-1'),
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented_ROI','RandomBrightnessContrast,ratio=(1.4,0.3,1.3,2.6,4.0,6.8)-1'),
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented_ROI','RandomGamma,ratio=(1.4,0.3,1.3,2.6,4.0,6.8)-1'),
            os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented_ROI','Rotate,ratio=(1.4,0.3,1.3,2.6,4.0,6.8)-2'),
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented_ROI','ShiftScaleRotate,ratio=(1.4,0.3,1.3,2.6,4.0,6.8)-1'),
            # os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented_ROI','Rotate,HorizontalFlip,RandomBrightnessContrast,GaussNoise,ElasticTransform,ShiftScaleRotate,RandomGamma,ratio=(1.4,0.3,1.3,2.6,4.0,6.8)-1'),
            os.path.join(os.curdir, "data", 'breast', 'cla', 'augmented_USG_ROI','Mixup,ratio=(8.9,0.4,8.3,15.7,23.2,37.0)-1'),
                                

        ]
    )

@dataclass
class FeaAugmentedDatasetConfig:
    pass


@dataclass
class BoundaryAugmentedDatasetConfig:
    pass

@dataclass
class CalcificationAugmentedDatasetConfig:
    pass

@dataclass
class DirectionAugmentedDatasetConfig:
    pass

@dataclass
class ShapeAugmentedDatasetConfig:
    pass


@dataclass
class ClaSingleFoldDatasetConfig(ClaDatasetConfig,ClaAugmentedDatasetConfig):
    _target_: str = "utils.BreastDataset.getBreastTrainValidData"


@dataclass
class ClaCrossValidationDatasetConfig(ClaDatasetConfig,ClaAugmentedDatasetConfig):
    _target_: str = "utils.BreastDataset.BreastCrossValidationData"
    k_fold:int=5
    

@dataclass
class FeaSingleFoldDatasetConfig(FeaDatasetConfig, FeaAugmentedDatasetConfig):
    _target_: str = "utils.BreastDataset.getBreastTrainValidData"

@dataclass
class FeaCrossValidationDatasetConfig(FeaDatasetConfig,FeaAugmentedDatasetConfig):
    _target_: str = "utils.BreastDataset.BreastCrossValidationData"


@dataclass
class BoundarySingleFoldDatasetConfig(BoundaryDatasetConfig, BoundaryAugmentedDatasetConfig):
    feature: str = 'boundary'
    ratio: List[int] = field(default_factory=lambda:[1,1])
    _target_: str = "utils.BreastDataset.getBreastTrainValidData"


@dataclass
class BoundaryCrossValidationDatasetConfig(BoundaryDatasetConfig, BoundaryAugmentedDatasetConfig):
    _target_: str = "utils.BreastDataset.BreastCrossValidationData"


@dataclass
class CalcificationSingleFoldDatasetConfig(CalcificationDatasetConfig, CalcificationAugmentedDatasetConfig):
    feature: str = 'calcification'
    ratio: List[int] = field(default_factory=lambda:[1,1])
    _target_: str = "utils.BreastDataset.getBreastTrainValidData"


@dataclass
class CalcificationCrossValidationDatasetConfig(CalcificationDatasetConfig, CalcificationAugmentedDatasetConfig):
    _target_: str = "utils.BreastDataset.BreastCrossValidationData"


@dataclass
class DirectionSingleFoldDatasetConfig(DirectionDatasetConfig, DirectionAugmentedDatasetConfig):
    feature: str = 'direction'
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
