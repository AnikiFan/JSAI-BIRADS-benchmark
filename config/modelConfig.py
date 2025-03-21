from dataclasses import dataclass, field
from omegaconf import MISSING
from typing import *

"""
模型配置，用于直接实例化模型
"""

@dataclass
class UnpretrainedModelConfig:
    pretrained: bool = False


@dataclass
class PretrainedModelConfig:
    pretrained: bool = False


@dataclass
class UnetModelConfig(PretrainedModelConfig):
    """
    _target_使得这个类的配置可以用于实例化target所指的可调用对象，
    该类的其他配置都会作为该可调用对象的参数
    该可调用函数需要在该配置文件中import
    也可以将target写为__main__.<callable>
    需要在运行的py文件中import可调用对象
    """
    _target_: str = "models.UnetClassifer.unet.UnetClassifier"
    in_channels: int = 3
    backbone: str = "resnet50"
    lr:float=0.001

@dataclass
class AlexNetModelConfig(UnpretrainedModelConfig):
    _target_:str = "models.model4compare.AlexNet.AlexNet"
    lr:float=0.001


@dataclass
class GoogleNetModelConfig(UnpretrainedModelConfig):
    _target_: str = "models.model4compare.GoogleNet.GoogleNet"
    lr:float=0.1


@dataclass
class NiNModelConfig(UnpretrainedModelConfig):
    _target_: str = "models.model4compare.NiN.NiN"
    lr:float=0.01

@dataclass
class VGGModelConfig(UnpretrainedModelConfig):
    _target_: str = "models.model4compare.VGG.VGG"
    lr:float=0.01
    arch:List[List[int]] = field(default_factory=lambda:[[1, 64], [1, 128], [2, 256], [2, 512], [2, 512]])

@dataclass
class LinearSanityCheckerModelConfig(UnpretrainedModelConfig):
    _target_: str = "models.model4compare.SanityChecker.LinearSanityChecker"
    lr:float=0.001

@dataclass
class ConvSanityCheckerModelConfig(UnpretrainedModelConfig):
    _target_: str = "models.model4compare.SanityChecker.ConvSanityChecker"
    lr:float=0.001

@dataclass
class PretrainedResNetModelConfig(PretrainedModelConfig):
    _target_:str="models.model4compare.ResNet18.ResNet18"
    num_classes:int=MISSING
    lr:float=0.001


@dataclass
class ResNetClassifierModelConfig(PretrainedModelConfig):
    _target_:str="models.PretrainClassifer.resnet.ResNetClassifier"
    num_classes:int=MISSING
    freeze_backbone:bool=False
    dropout:float=0.2
    lr:float=0.001
    

@dataclass 
class MobileNetModleConfig(PretrainedModelConfig):
    _target_:str="models.MobileNet.MobileNet_V2.MyMobileNetV2"
    lr:float=0.001
