from dataclasses import dataclass, field
from omegaconf import MISSING
from pathlib import Path
from models.UnetClassifer.unet import UnetClassifier
from models.model4compare.AlexNet import AlexNet
from models.model4compare.GoogleNet import GoogleNet
from models.model4compare.NiN import NiN
from models.model4compare.VGG import VGG
from models.model4compare.SanityChecker import LinearSanityChecker,ConvSanityChecker
from typing import *

"""
模型配置，用于直接实例化模型，所以不能有除模型实例化所需参数以外的配置项
"""


@dataclass
class DefaultModelConfig:
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
    pretrained: bool = True

@dataclass
class AlexNetModelConfig:
    _target_:str = "models.model4compare.AlexNet.AlexNet"
    lr:float=0.001


@dataclass
class GoogleNetModelConfig:
    _target_: str = "models.model4compare.GoogleNet.GoogleNet"
    lr:float=0.1


@dataclass
class NiNModelConfig:
    _target_: str = "models.model4compare.NiN.NiN"
    lr:float=0.01


@dataclass
class VGGModelConfig:
    _target_: str = "models.model4compare.VGG.VGG"
    lr:float=0.01
    arch:List[List[int]] = field(default_factory=lambda:[[1, 64], [1, 128], [2, 256], [2, 512], [2, 512]])

@dataclass
class LinearSanityCheckerModelConfig:
    _target_: str = "models.model4compare.SanityChecker.LinearSanityChecker"
    lr:float=0.001

@dataclass
class ConvSanityCheckerModelConfig:
    _target_: str = "models.model4compare.SanityChecker.ConvSanityChecker"
    lr:float=0.001
