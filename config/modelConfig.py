from dataclasses import dataclass, field
from omegaconf import MISSING
from pathlib import Path
from models.UnetClassifer.unet import UnetClassifier

@dataclass
class DefaultModelConfig:
    _target_: str = "models.UnetClassifer.unet.UnetClassifier"
    num_classes: int=6
    in_channels: int=3
    backbone: str="resnet50"
    pretrained : bool = True