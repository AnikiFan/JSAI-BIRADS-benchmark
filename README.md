from hydra.core.config_store import ConfigStore

# 训练框架说明

## 概述

本训练框架主要由三部分组成：

1. hydra超参数配置
2. Trainer类
3. TableDataset类

### hydra超参数配置

为了使用由hydra库配置的参数，需要在程序运行之初调用以下语句

```python
cs = ConfigStore.instance()
cs.store(name="config",node=Config)
```

为了在某个函数中使用Config中存储的超参数，要使用hydra.main装饰器
```python
import hydra
@hydra.main(version_base=None,config_name="config")
def foo(cfg:Config):
  pass
```

```python
cs.store(group='train', name="default", node=DefaultTrainConfig)
cs.store(group='model', name="default", node=DefaultModelConfig)
cs.store(group='dataset', name="multiple", node=CrossValidationDatasetConfig)
cs.store(group='optimizer', name="default", node=DefaultOptimizerConfig)
cs.store(group='env', name="fx", node=FXEnvConfig)
cs.store(group='train_transform', name="default", node=DefaultTrainTransformConfig)
cs.store(group='valid_transform', name="default", node=DefaultValidTransformConfig)
```

本框架将超参数分为以下类

1. 环境配置
> cpu，gpu，路径等相关配置
2. 训练配置
> epoch数、早停等相关配置
3. 数据集配置
> num_classes等用于实例化数据集类的参数配置
4. 模型配置
> 用于实例化model类的参数配置
5. 优化器配置
> 用于实例化优化器的参数配置
6. 训练集变换配置
> 用于实例化训练集所用变换的参数配置
7. 验证集变换配置
> 用于实例化验证集所用变换的参数配置



# tensorboard使用方法

```commandline
tensorboard --logdir=./runs
```

# 仓库目录结构

```
卷 游戏及文件 的文件夹 PATH 列表
卷序列号为 EA13-00A1
D:.
├─.idea
│  ├─inspectionProfiles
│  └─shelf
│      └─Uncommitted_changes_before_Update_at_9_21_2024_7_56_PM_[Changes]
├─data
│  ├─breast
│  │  ├─.ipynb_checkpoints
│  │  ├─.jupyter
│  │  │  └─desktop-workspaces
│  │  ├─BUS
│  │  │  ├─.idea
│  │  │  │  └─inspectionProfiles
│  │  │  ├─Images
│  │  │  └─Masks
│  │  ├─myTrain
│  │  │  └─cla
│  │  ├─OASBUD
│  │  ├─testA
│  │  │  └─cla
│  │  ├─test_A
│  │  │  ├─cla
│  │  │  │  ├─2类
│  │  │  │  │  ├─images
│  │  │  │  │  └─labels
│  │  │  │  ├─3类
│  │  │  │  │  ├─images
│  │  │  │  │  └─labels
│  │  │  │  ├─4A类
│  │  │  │  │  ├─images
│  │  │  │  │  └─labels
│  │  │  │  ├─4B类
│  │  │  │  │  ├─images
│  │  │  │  │  └─labels
│  │  │  │  ├─4C类
│  │  │  │  │  ├─images
│  │  │  │  │  └─labels
│  │  │  │  └─5类
│  │  │  │      ├─images
│  │  │  │      └─labels
│  │  │  └─fea
│  │  │      ├─boundary_labels
│  │  │      ├─calcification_labels
│  │  │      ├─direction_labels
│  │  │      ├─images
│  │  │      └─shape_labels
│  │  ├─train
│  │  │  ├─cla
│  │  │  │  ├─2类
│  │  │  │  │  ├─images
│  │  │  │  │  └─labels
│  │  │  │  ├─3类
│  │  │  │  │  ├─images
│  │  │  │  │  └─labels
│  │  │  │  ├─4A类
│  │  │  │  │  ├─images
│  │  │  │  │  └─labels
│  │  │  │  ├─4B类
│  │  │  │  │  ├─images
│  │  │  │  │  └─labels
│  │  │  │  ├─4C类
│  │  │  │  │  ├─images
│  │  │  │  │  └─labels
│  │  │  │  └─5类
│  │  │  │      ├─images
│  │  │  │      └─labels
│  │  │  └─fea
│  │  │      ├─boundary_labels
│  │  │      ├─calcification_labels
│  │  │      ├─direction_labels
│  │  │      ├─images
│  │  │      └─shape_labels
│  │  ├─train_valid_test
│  │  │  ├─test
│  │  │  │  └─unknown
│  │  │  ├─train
│  │  │  │  ├─2类
│  │  │  │  ├─3类
│  │  │  │  ├─4A类
│  │  │  │  ├─4B类
│  │  │  │  ├─4C类
│  │  │  │  └─5类
│  │  │  ├─train_valid
│  │  │  │  ├─2类
│  │  │  │  ├─3类
│  │  │  │  ├─4A类
│  │  │  │  ├─4B类
│  │  │  │  ├─4C类
│  │  │  │  └─5类
│  │  │  └─valid
│  │  │      ├─2类
│  │  │      ├─3类
│  │  │      ├─4A类
│  │  │      ├─4B类
│  │  │      ├─4C类
│  │  │      └─5类
│  │  └─USG
│  │      └─.idea
│  │          └─inspectionProfiles
│  ├─FashionMNIST
│  │  └─raw
│  └─test
├─models
│  ├─model4compare
│  │  └─__pycache__
│  └─UnetClassifer
│      └─__pycache__
├─TDSNet
│  └─__pycache__
├─test
└─utils
    ├─MyBlock
    │  └─__pycache__
    └─__pycache__
```




# train保存的模型文件结构
```
checkPoint
.
├── Unet_Breast_20240911_163649
│   ├── cfg.json
│   ├── model
│   │   ├── Unet_ac0.386731_f10.232353_0.pth
│   │   └── Unet_ac0.422735_f10.289592_1.pth
│   ├── model_cfg.json
│   ├── optimizer
│   │   ├── Unet_ac0.386731_f10.232353_0.pth
│   │   └── Unet_ac0.422735_f10.289592_1.pth
│   └── transforms_cfg.json
└── readme.txt
```


# 新增文件夹说明：

- **checkPoint** ： 保存训练过程中的模型文件
- model_data : Unet中用来下载预训练模型的地址，建议其他模型与之一致
- **models** ： 编写的模型文件
- utils ： 一些工具函数，如检查数据集，早停，输出过滤（过滤f1wanning）等
  - MyModel ： 一些神经网络中所用的模块（pytorch中没有现成的）
