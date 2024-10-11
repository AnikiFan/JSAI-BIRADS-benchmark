from dataclasses import dataclass, field
from typing import *
from utils.MyCrop import MyCrop
import timm
from config.modelConfig import ViTClassifierModelConfig
from torchvision.transforms import InterpolationMode
from torchvision import transforms
"""
图像变换配置
"""


@dataclass
class DefaultTrainTransformConfig:
    _target_: str = "torchvision.transforms.Compose"
    transforms: List[Any] = field(
        default_factory=lambda: [MyCropConfig(), ResizeConfig()]
    )


@dataclass
class DefaultValidTransformConfig:
    _target_: str = "torchvision.transforms.Compose"
    transforms: List[Any] = field(
        default_factory=lambda: [MyCropConfig(), ResizeConfig()]
    )

@dataclass
class Transform_RandomResizedCropConfig:
    _target_: str = "torchvision.transforms.RandomResizedCrop"
    size: List[int] = field(default_factory=lambda: [224, 224])
    scale: Tuple[float, float] = field(default_factory=lambda: (0.08, 1.0))
    ratio: Tuple[float, float] = field(default_factory=lambda: (0.75, 1.3333))
    interpolation: InterpolationMode = InterpolationMode.BICUBIC
    _convert_: str = "all"
    
@dataclass 
class Transform_RandomHorizontalFlipConfig:
    _target_: str = "torchvision.transforms.RandomHorizontalFlip"
    p: float = 0.5
    _convert_: str = "all"
    
@dataclass
class Transform_ColorJitterConfig:
    _target_: str = "torchvision.transforms.ColorJitter"
    brightness: Tuple[float, float] = field(default_factory=lambda: (0.6, 1.4))
    contrast: Tuple[float, float] = field(default_factory=lambda: (0.6, 1.4))
    saturation: Tuple[float, float] = field(default_factory=lambda: (0.6, 1.4))
    _convert_: str = "all"
    

@dataclass
class Transform_ToTensorConfig:
    _target_: str = "torchvision.transforms.ToTensor"
    _convert_: str = "all"

@dataclass
class Transform_NormalizeConfig:
    _target_: str = "torchvision.transforms.Normalize"
    mean: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    std: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    _convert_: str = "all"



@dataclass
class ToTensorConfig:
    _target_: str = "torchvision.transforms.ToTensor"


@dataclass
class MyCropConfig:
    _target_: str = "utils.MyCrop.MyCrop"


@dataclass
class ResizeConfig:
    """
    这里设置_convert_="all"是为了让size在传入参数时变为list类型，否则会以hydra库中的类传入，不符合规定
    注意，convert只支持转换为list，不支持转换为tuple
    """

    _target_: str = "torchvision.transforms.Resize"
    size: List[int] = field(default_factory=lambda: [224, 224])
    antialias: bool = True  # 显式设置为True，避免警告，抗锯齿
    _convert_: str = "all"


@dataclass
class PILResizeConfig:
    _target_: str = "utils.PILResize.PILResize"
    size: List[int] = field(default_factory=lambda: [128, 128])
    _convert_: str = "all"

# @dataclass


@dataclass
class NormalizeConfig:
    _target_: str = "torchvision.transforms.Normalize"
    mean: List[float] = field(default_factory=lambda: [0.4914, 0.4822, 0.4465])
    std: List[float] = field(default_factory=lambda: [0.2023, 0.1994, 0.201])


@dataclass
class CustomTrainTransformConfig:
    _target_: str = "torchvision.transforms.Compose"
    transforms: List[Any] = field(
        default_factory=lambda: [
            MyCropConfig(),
            ResizeConfig(),
            # ToTensorConfig(),
            NormalizeConfig(),
        ]
    )


@dataclass
class CustomValidTransformConfig:
    _target_: str = "torchvision.transforms.Compose"
    transforms: List[Any] = field(
        default_factory=lambda: [
            MyCropConfig(),
            ResizeConfig(),
            # ToTensorConfig(),
            NormalizeConfig(),
        ]
    )


# @dataclass
# class TimmTrainTransformConfig:
#     _target_: str = "timm.data.create_transform"
#     is_training: bool = True
#     transforms: List[Any] = field(default_factory=lambda: [
#         MyCropConfig(),
#         ResizeConfig(),
#         # ToTensorConfig(),
#         NormalizeConfig()
#     ])


# @dataclass
# class ViTClassifierModelDataConfig:
#     _target_: str = "timm.data.resolve_model_data_config"
#     model_name: str = field(
#         default_factory=lambda: ViTClassifierModelConfig().model_name
#     )


# @dataclass
# class ViTClassifierCreateTransformConfig:
#     _target_: str = "timm.data.create_transform"
#     is_training: bool = True



@dataclass
class ViTClassifierTransformConfig:
    _target_: str = 'torchvision.transforms.Compose'
    transforms: List[Any] = field(
        default_factory=lambda: [
        MyCropConfig(),
        # 随机裁剪并调整图像大小，使用双三次插值
        Transform_RandomResizedCropConfig(),
        # 随机水平翻转
        Transform_RandomHorizontalFlipConfig(),
        # 调整图像的亮度、对比度和饱和度
        Transform_ColorJitterConfig(),
        # 将图像转换为张量
        # Transform_ToTensorConfig(),
        Transform_NormalizeConfig()
        ]
    )
    _convert_: str = "all"

