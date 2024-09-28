from dataclasses import dataclass, field
from omegaconf import MISSING
from pathlib import Path
import os
@dataclass
class DefaultEnvConfig:
    data_folder_path:str=os.path.join(os.curdir,'data')