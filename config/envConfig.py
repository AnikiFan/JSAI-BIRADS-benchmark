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
    pin_memory_device: str = device if device.startswith('cuda') else ''
    data_folder_path: Path = os.path.join(os.curdir, 'data')
    model_weight_path:Path = os.path.join(data_folder_path, 'model_weight')

@dataclass
class FXEnvConfig(EnvConfig):
    pass


@dataclass
class ZHYLocalEnvConfig(EnvConfig):
    # device: str = getDevice()
    device: str = 'cpu'
    pin_memory: bool = False
    
class ZhyRemoteEnvConfig(EnvConfig):
    pass
    


@dataclass
class YZLEnvConfig(EnvConfig):
    pass
