from dataclasses import dataclass, field
from typing import *
from utils.MyCrop import MyCrop

"""
图像变换配置
"""


@dataclass
class DefaultTrainTransformConfig:
    _target_: str = "torchvision.transforms.Compose"
    transforms: List[Any] = field(default_factory=lambda: [ MyCropConfig(),ResizeConfig()])


@dataclass
class DefaultValidTransformConfig:
    _target_: str = "torchvision.transforms.Compose"
    transforms: List[Any] = field(default_factory=lambda: [MyCropConfig(), ResizeConfig()])


@dataclass
class ToTensorConfig:
    _target_: str = "torchvision.transforms.ToTensor"


@dataclass
class MyCropConfig:
    _target_: str = "utils.MyCrop.MyCrop"


@dataclass
class ResizeConfig:
    """
    这里设置_convert_="all"是为了让size在传入参数是变为list类型，否则会以hydra库中的类传入，不符合规定
    注意，conver只支持转换为list，不支持转换为tuple
    """
    _target_: str = "torchvision.transforms.Resize"
    size: List[int] = field(default_factory=lambda: [256, 256])
    antialias: bool = True # 显式设置为True，避免警告，抗锯齿
    _convert_: str = "all"


@dataclass
class PILResizeConfig:
    _target_: str = "utils.PILResize.PILResize"
    size: List[int] = field(default_factory=lambda: [128, 128])
    _convert_: str = "all"


@dataclass
class NormalizeConfig:
    _target_: str = "torchvision.transforms.Normalize"
    mean: List[float] = field(default_factory=lambda: [0.4914, 0.4822, 0.4465])
    std: List[float] = field(default_factory=lambda: [0.2023, 0.1994, 0.201])


@dataclass
class CustomTrainTransformConfig:
    _target_: str = "torchvision.transforms.Compose"
    transforms: List[Any] = field(default_factory=lambda: [
        MyCropConfig(),
        ResizeConfig(),
        # ToTensorConfig(),
        NormalizeConfig()
    ])


@dataclass
class CustomValidTransformConfig:
    _target_: str = "torchvision.transforms.Compose"
    transforms: List[Any] = field(default_factory=lambda: [
        MyCropConfig(),
        ResizeConfig(),
        # ToTensorConfig(),
        NormalizeConfig()
    ])
