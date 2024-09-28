from dataclasses import dataclass, field
from omegaconf import MISSING
from pathlib import Path
from utils.ClaDataset import getClaTrainValidData
import os
@dataclass
class DefaultDatasetConfig:
    _target_:str="utils.ClaDataset.getClaTrainValidData"
    data_folder_path:Path=Path(os.path.join(os.curdir,'data'))
