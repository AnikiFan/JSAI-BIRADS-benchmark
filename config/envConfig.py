from dataclasses import dataclass, field
from omegaconf import MISSING
from pathlib import Path
import os
"""
环境配置，如路径等
"""
@dataclass
class DefaultEnvConfig:
    data_folder_path:str=os.path.join(os.curdir,'data')