from dataclasses import dataclass, field
from pathlib import Path
import os
from utils.tools import getDevice

"""
环境配置，如路径等
"""


@dataclass
class EnvConfig:
    device: str = getDevice()
    pin_memory: bool = True


@dataclass
class FXEnvConfig(EnvConfig):
    data_folder_path: Path = os.path.join(os.curdir, 'data')


@dataclass
class ZHYLocalEnvConfig(EnvConfig):
    data_folder_path: Path = os.path.join(os.curdir, 'data')
    # device: str = getDevice()
    device: str = 'cpu'
    pin_memory: bool = False
    
@dataclass
class ZhyRemoteEnvConfig(EnvConfig):
    data_folder_path: Path = os.path.join(os.curdir, 'data')
    # device: str = 'cuda'
    # pin_memory: bool = True


@dataclass
class YZLEnvConfig(EnvConfig):
    pass
