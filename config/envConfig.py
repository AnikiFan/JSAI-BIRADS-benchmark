from dataclasses import dataclass, field
from omegaconf import MISSING
from pathlib import Path
import os
from utils.tools import getDevice

"""
环境配置，如路径等
"""


@dataclass
class EnvConfig:
    device: str = getDevice()
    pin_memory: bool = getDevice() == "cuda"


@dataclass
class FXEnvConfig(EnvConfig):
    data_folder_path: Path = os.path.join(os.curdir, 'data')


@dataclass
class ZHYEnvConfig(EnvConfig):
    pass


@dataclass
class YZLEnvConfig(EnvConfig):
    pass
