from dataclasses import dataclass, field
from omegaconf import MISSING
from typing import *

"""
模型配置，用于直接实例化模型
"""

@dataclass
class UnpretrainedModelConfig:
    pretrained: bool = False


@dataclass
class PretrainedModelConfig:
    pretrained: bool = True


@dataclass
class UnetModelConfig(PretrainedModelConfig):
    """
    _target_使得这个类的配置可以用于实例化target所指的可调用对象，
    该类的其他配置都会作为该可调用对象的参数
    该可调用函数需要在该配置文件中import
    也可以将target写为__main__.<callable>
    需要在运行的py文件中import可调用对象
    """
    _target_: str = "models.UnetClassifer.unet.UnetClassifier"
    in_channels: int = 3
    backbone: str = "resnet50"
    lr:float=0.001

@dataclass
class AlexNetModelConfig(UnpretrainedModelConfig):
    _target_:str = "models.model4compare.AlexNet.AlexNet"
    lr:float=0.001


@dataclass
class GoogleNetModelConfig(UnpretrainedModelConfig):
    _target_: str = "models.model4compare.GoogleNet.GoogleNet"
    lr:float=0.1


@dataclass
class NiNModelConfig(UnpretrainedModelConfig):
    _target_: str = "models.model4compare.NiN.NiN"
    lr:float=0.01

@dataclass
class VGGModelConfig(UnpretrainedModelConfig):
    _target_: str = "models.model4compare.VGG.VGG"
    lr:float=0.01
    arch:List[List[int]] = field(default_factory=lambda:[[1, 64], [1, 128], [2, 256], [2, 512], [2, 512]])

@dataclass
class LinearSanityCheckerModelConfig(UnpretrainedModelConfig):
    _target_: str = "models.model4compare.SanityChecker.LinearSanityChecker"
    lr:float=0.001

@dataclass
class ConvSanityCheckerModelConfig(UnpretrainedModelConfig):
    _target_: str = "models.model4compare.SanityChecker.ConvSanityChecker"
    lr:float=0.001

@dataclass
class PretrainedResNetModelConfig(PretrainedModelConfig):
    _target_:str="models.model4compare.ResNet18.ResNet18"
    num_classes:int=MISSING
    lr:float=0.001


@dataclass
class ResNetClassifierModelConfig(PretrainedModelConfig):
    _target_:str="models.PretrainClassifer.resnet.ResNetClassifier"
    num_classes:int=MISSING
    # feature_only:bool=True
    # freeze_backbone:bool=True
    dropout:float=0.2
    lr:float=0.001
    

@dataclass 
class MobileNetModleConfig(PretrainedModelConfig):
    _target_:str="models.MobileNet.MobileNet_V2.MyMobileNetV2"
    lr:float=0.001


@dataclass
class PretrainedClassifierModelConfig:
    _target_:str="models.UnetClassifer.unet.PretrainedClassifier"
    resnet_type:str="resnet50"
    num_classes:int=MISSING
    pretrained:bool=True
    backbone:str="resnet50"
    freeze_backbone:bool=False
    # dropout:float=0.2
    lr:float=0.001
    
    
    
@dataclass
class MobileNetV2ClassifierModelConfig(PretrainedModelConfig):
    _target_:str="models.MobileNet.MobileNet_V2_Classifier.MobileNetV2Classifier"
    num_classes:int=MISSING
    feature_extract:bool=False
    lr:float=0.01
    
@dataclass
class ViTClassifier_timm_ModelConfig(PretrainedModelConfig):
    _target_:str="models.timm.ViT.ViTClassifier_timm"
    # model_name:str="vit_base_patch16_224.augreg_in21k"
    # model_name:str='vit_mediumd_patch16_reg4_gap_256.sbb_in12k_ft_in1k'
    model_name:str='fastvit_ma36.apple_in1k'
    # model_name:str='densenet121.ra_in1k'
    classifier_head:bool=False
    num_classes:int=6
    pretrained:bool=True
    lr:float=0.001
    drop_rate:float=0.0
    drop_path_rate:float=0.1
    

@dataclass
class DenseNetClassifier_timm_ModelConfig(PretrainedModelConfig):
    _target_:str="models.timm.denseNet.DenseNetClassifier"
    lr:float=0.001
    features_only:bool=False
    num_classes:int=6
    pretrained:bool=True
    freeze_backbone:bool=False
