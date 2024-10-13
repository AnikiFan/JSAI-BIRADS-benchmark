from dataclasses import dataclass, field
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore
from hydra.conf import HydraConf
from .trainConfig import *
from .datasetConfig import *
from .modelConfig import *
from .optimizerConfig import *
from .envConfig import *
from .transformConfig import *
from .schedularConfig import *
from typing import *

cla_defaults = [
        {"train": "cla_task"},
        {"model": "linear_sanity_check"},
        {"dataset": "cla_single"},
        {"optimizer": "SGD"},
        {"env": "fx"},
        {"train_transform": "default"},
        {"valid_transform": "default"},
        {"schedular": "exponential"},
    ]

fea_defaults = [
        {"train": "fea_task"},
        {"model": "mobilenet_v2"},
        {"dataset": "fea_single"},
        {"optimizer": "SGD"},
        {"env": "fx"},
        {"train_transform": "default"},
        {"valid_transform": "default"},
        {"schedular": "exponential"},
    ]

boundary_defaults = [
        {"train": "boundary_task"},
        {"model": "linear_sanity_check"},
        {"dataset": "boundary_single"},
        {"optimizer": "SGD"},
        {"env": "fx"},
        {"train_transform": "default"},
        {"valid_transform": "default"},
        {"schedular": "exponential"},
    ]

calcification_defaults = [
        {"train": "calcification_task"},
        {"model": "linear_sanity_check"},
        {"dataset": "calcification_single"},
        {"optimizer": "SGD"},
        {"env": "fx"},
        {"train_transform": "default"},
        {"valid_transform": "default"},
        {"schedular": "exponential"},
    ]

direction_defaults = [
        {"train": "direction_task"},
        {"model": "linear_sanity_check"},
        {"dataset": "direction_single"},
        {"optimizer": "SGD"},
        {"env": "fx"},
        {"train_transform": "default"},
        {"valid_transform": "default"},
        {"schedular": "exponential"},
    ]

shape_defaults = [
        {"train": "shape_task"},
        {"model": "linear_sanity_check"},
        {"dataset": "shape_single"},
        {"optimizer": "SGD"},
        {"env": "fx"},
        {"train_transform": "default"},
        {"valid_transform": "default"},
        {"schedular": "exponential"},
    ]

@dataclass
class Config:
    defaults: List[Any] = field(default_factory=lambda:boundary_defaults)
    train: Any = MISSING
    model: Any = MISSING
    dataset: Any = MISSING
    optimizer: Any = MISSING
    env: Any = MISSING
    train_transform: Any = MISSING
    valid_transform: Any = MISSING
    schedular: Any = MISSING


def init_config():
    """
    初始化配置
    :return:
    """
    # 初始化
    cs = ConfigStore.instance()

    cs.store(group='train', name="cla_task", node=ClaTrainConfig)
    cs.store(group='train', name="fea_task", node=FeaTrainConfig)
    cs.store(group='train', name="boundary_task", node=BoundaryTrainConfig)
    cs.store(group='train', name="calcification_task", node=CalcificationTrainConfig)
    cs.store(group='train', name="direction_task", node=DirectionTrainConfig)
    cs.store(group='train', name="shape_task", node=ShapeTrainConfig)
    cs.store(group='train', name="cla_task_remote", node=RemoteTrainConfig)

    cs.store(group='model', name="alex_net", node=AlexNetModelConfig)
    cs.store(group='model', name="google_net", node=GoogleNetModelConfig)
    cs.store(group='model', name="NiN", node=NiNModelConfig)
    cs.store(group='model', name="VGG", node=VGGModelConfig)
    cs.store(group='model', name="linear_sanity_check", node=LinearSanityCheckerModelConfig)
    cs.store(group='model', name="conv_sanity_check", node=ConvSanityCheckerModelConfig)
    cs.store(group='model', name="pretrained_resnet", node=PretrainedResNetModelConfig)
    cs.store(group='model', name="unet_classifier", node=UnetModelConfig)
    cs.store(group='model', name="resnet_classifier", node=ResNetClassifierModelConfig)
    cs.store(group='model', name="mobilenet_v2", node=MobileNetModleConfig)
    cs.store(group='model', name="pretrained_classifier", node=PretrainedClassifierModelConfig)
    cs.store(group='model', name="mobilenet_v2_classifier", node=MobileNetV2ClassifierModelConfig)
    cs.store(group='model', name="vit_classifier_timm", node=ViTClassifier_timm_ModelConfig)
    cs.store(group='model', name="denseNet_classifier_timm", node=DenseNetClassifier_timm_ModelConfig)

    cs.store(group='dataset', name="fashion_mnist", node=FashionMNISTDatasetConfig)
    cs.store(group='dataset', name="mnist", node=MNISTDatasetConfig)
    cs.store(group='dataset', name="cifar10", node=CIFAR10DatasetConfig)
    cs.store(group='dataset', name="cla_single", node=ClaSingleFoldDatasetConfig)
    cs.store(group='dataset', name="cla_multiple", node=ClaCrossValidationDatasetConfig)
    cs.store(group='dataset', name="fea_single", node=FeaSingleFoldDatasetConfig)
    cs.store(group='dataset', name="fea_multiple", node=FeaCrossValidationDatasetConfig)
    cs.store(group='dataset', name="boundary_single", node=BoundarySingleFoldDatasetConfig)
    cs.store(group='dataset', name="boundary_multiple", node=BoundaryCrossValidationDatasetConfig)
    cs.store(group='dataset', name="calcification_single", node=CalcificationSingleFoldDatasetConfig)
    cs.store(group='dataset', name="calcification_multiple", node=CalcificationCrossValidationDatasetConfig)
    cs.store(group='dataset', name="direction_single", node=DirectionSingleFoldDatasetConfig)
    cs.store(group='dataset', name="direction_multiple", node=DirectionCrossValidationDatasetConfig)
    cs.store(group='dataset', name="shape_single", node=ShapeSingleFoldDatasetConfig)
    cs.store(group='dataset', name="shape_multiple", node=ShapeCrossValidationDatasetConfig)

    cs.store(group='optimizer', name="SGD", node=SGDOptimizerConfig)
    cs.store(group='optimizer', name="Adam", node=AdamOptimizerConfig)
    cs.store(group='optimizer', name="AdamW", node=AdamWOptimizerConfig)

    cs.store(group='env', name="fx", node=FXEnvConfig)
    cs.store(group='env', name="zhy_local", node=ZHYLocalEnvConfig)
    cs.store(group='env', name="zhy_remote", node=ZhyRemoteEnvConfig)
    cs.store(group='env', name="yzl", node=YZLEnvConfig)

    cs.store(group='train_transform', name="default", node=DefaultTrainTransformConfig)
    cs.store(group='train_transform', name="custom", node=CustomTrainTransformConfig)
    cs.store(group='train_transform', name="vit", node=ViTClassifierTrainTransformConfig)
    cs.store(group='train_transform', name="fastvit", node=FastViT_trainTransformConfig)
    cs.store(group='train_transform', name="empty", node=EmptyTransformConfig)

    cs.store(group='valid_transform', name="default", node=DefaultValidTransformConfig)
    cs.store(group='valid_transform', name="custom", node=CustomValidTransformConfig)
    cs.store(group='valid_transform', name="vit", node=ViTClassifierValidTransformConfig)
    cs.store(group='valid_transform', name="fastvit", node=FastViT_validTransformConfig)
    cs.store(group='valid_transform', name="empty", node=EmptyTransformConfig)

    cs.store(group='schedular', name="exponential", node=ExponentialLRConfig)
    cs.store(group='schedular', name="dummy", node=DummySchedularConfig)

    # 初始化
    cs.store(name="my_config", node=Config)
