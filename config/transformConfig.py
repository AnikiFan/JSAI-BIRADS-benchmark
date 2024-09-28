from dataclasses import dataclass, field
from omegaconf import MISSING
from pathlib import Path
from typing import *
from torchvision.transforms import Compose,ToTensor
from utils.MyBlock.MyCrop import MyCrop
from utils.PILResize import PILResize

@dataclass
class DefaultTrainTransformConfig:
    _target_: str = "torchvision.transforms.Compose"
    transforms:List[Any] = field(default_factory= lambda:[MyCropConfig(),PILResizeConfig()])

@dataclass
class DefaultValidTransformConfig:
    _target_: str = "torchvision.transforms.Compose"
    transforms:List[Any] = field(default_factory= lambda:[MyCropConfig(),PILResizeConfig()])

@dataclass
class ToTensorConfig:
    _target_: str = "torchvision.transforms.ToTensor"

@dataclass
class MyCropConfig:
    _target_: str = "utils.MyBlock.MyCrop.MyCrop"

@dataclass
class ResizeConfig:
    _target_: str = "torchvision.transforms.ToTensor"
    size: Tuple[int] = (400,400)

@dataclass
class PILResizeConfig:
    _target_: str = "utils.PILResize.PILResize"
    size: Tuple[int] = (128,128)
