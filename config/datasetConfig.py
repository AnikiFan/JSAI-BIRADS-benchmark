from dataclasses import dataclass, field
from omegaconf import MISSING
from pathlib import Path
from utils.ClaDataset import getClaTrainValidData,ClaCrossValidationData
import os
"""
数据集配置，用于直接实例化数据集，所以不能有数据集所需的参数以外的配置项
数据集对象应该是一个迭代器，每次迭代返回train_dataset和valid_dataset
涉及到路径的参数建议使用_partial_，在应用时再将环境配置中的路径传入
"""
@dataclass
class DefaultDatasetConfig:
    _target_:str="utils.ClaDataset.ClaCrossValidationData"
    data_folder_path:Path=Path(os.path.join(os.curdir,'data'))
    image_format:str="Tensor"
