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
    transforms:List[Any] = field(default_factory= lambda:[MyCropConfig(),ResizeConfig()])

@dataclass
class DefaultValidTransformConfig:
    _target_: str = "torchvision.transforms.Compose"
    transforms:List[Any] = field(default_factory= lambda:[MyCropConfig(),ResizeConfig()])

@dataclass
class ToTensorConfig:
    _target_: str = "torchvision.transforms.ToTensor"

@dataclass
class MyCropConfig:
    _target_: str = "utils.MyBlock.MyCrop.MyCrop"

@dataclass
class ResizeConfig:
    _target_: str = "torchvision.transforms.Resize"
    size: List[int] = field(default_factory=lambda:[256,256])
    _convert_:str="all"

@dataclass
class PILResizeConfig:
    _target_: str = "utils.PILResize.PILResize"
    size: List[int] = field(default_factory=lambda:[128,128])
    _convert_:str="all"
