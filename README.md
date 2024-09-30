from hydra.core.config_store import ConfigStore

# 训练框架说明

## 概述

本训练框架主要由三部分组成：

1. hydra超参数配置
2. Trainer类
3. 数据处理相关类

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

### 配置分组

本框架将超参数分为以下组

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

其中，数据集、模型、优化器、训练集变换和验证集变换配置都直接用于实例化相应的对象，例如用`instantiate(cfg.model)`来实例化训练所用的模型。

## `Trainer`类

Trainer类用于执行不同配置的训练任务。训练任务中所用到的训练集，模型，优化器，训练集和验证集变换都直接
通过配置实例化得到，因此，只需要更改配置文件即可执行不同的训练任务。

### 训练集

```python
    def train(self)->None:
        for train_ds, valid_ds in instantiate(self.cfg.dataset, data_folder_path=self.cfg.env.data_folder_path,
                                              train_transform=self.train_transform,
                                              valid_transform=self.valid_transform):
            loss, f1_score, accuracy, confusion_matrix = self.train_one_fold(
                DataLoader(train_ds, batch_size=self.cfg.train.batch_size, shuffle=True, pin_memory=True,
                           drop_last=False, num_workers=self.cfg.train.num_workers,
                           pin_memory_device=self.cfg.env.device),
                DataLoader(valid_ds, batch_size=self.cfg.train.batch_size, shuffle=True, pin_memory=True,
                           drop_last=False, num_workers=self.cfg.train.num_workers,
                           pin_memory_device=self.cfg.env.device)
            )
```
数据集在此处被实例化，唯一的要求是数据集类是迭代器类型，每次迭代返回训练集和验证集，并且训练集和验证集能够用于实例化`DataLoader`类(因此要求label应该是一个整体，不能拆分开来)。
如果不需要`data_folder_path`等参数，可以在参数列表中写上`**kwargs`来接收多余的关键字参数。

### 模型

```python
    @time_logger
    def train_one_fold(self, train_loader: DataLoader, valid_loader: DataLoader) -> Tuple[
        float, float, float, torch.Tensor]:
        """
        训练一折
        :param train_loader:
        :param valid_loader:
        :return: 该折训练中，在单个验证集上达到的最佳的指标
        """
        best_loss, best_f1, best_accuracy, best_confusion_matrix = 1_000_000., None, None, None
        model = instantiate(self.cfg.model)
        optimizer = instantiate(self.cfg.optimizer, params=model.parameters())
        writer = SummaryWriter(os.path.join('runs', self.make_writer_title()))
```
模型在此处实例化。
```python
 def train_one_epoch(self, *, model, train_loader: DataLoader, optimizer, epoch_index: int,
                        tb_writer: SummaryWriter) -> Tuple[float, float, float]:
        '''
        训练一个 epoch
        :param model: 模型
        :param epoch_index: 当前 epoch
        :param train_loader: 训练数据加载器
        :param num_class: 类别数量
        :param tb_writer: TensorBoard 写入器
        '''
        outputs, labels = [], []

        model.train(True)

        for i, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch_index}"), start=1):
            input, label = data
            input = input.to(self.cfg.env.device)
            label = label.to(self.cfg.env.device)
            optimizer.zero_grad()
            output = model(input)
            outputs.append(output)
    
```
model在上处被调用，因此，model需要有train成员函数，并且能够接收dataloader给出的input，label对中的input并
返回output。

### 优化器

```python
 @time_logger
    def train_one_fold(self, train_loader: DataLoader, valid_loader: DataLoader) -> Tuple[
        float, float, float, torch.Tensor]:
        """
        训练一折
        :param train_loader:
        :param valid_loader:
        :return: 该折训练中，在单个验证集上达到的最佳的指标
        """
        best_loss, best_f1, best_accuracy, best_confusion_matrix = 1_000_000., None, None, None
        model = instantiate(self.cfg.model)
        optimizer = instantiate(self.cfg.optimizer, params=model.parameters())
        writer = SummaryWriter(os.path.join('runs', self.make_writer_title()))
```
优化器在此处实例化，
```python
        for i, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch_index}"), start=1):
            input, label = data
            input = input.to(self.cfg.env.device)
            label = label.to(self.cfg.env.device)
            optimizer.zero_grad()
            output = model(input)
            outputs.append(output)
            labels.append(label)
            loss = self.loss_fn(output, label)
            loss.backward()
            optimizer.step()
```
优化器在上处被调用，因此自定义的优化器要支持这些功能

### 损失函数
```python
class Trainer:
    def __init__(self, cfg: Config):
        self.cfg: Config = cfg
        self.loss_fn = instantiate(cfg.train.loss_function)
```
损失函数在此实例化
```python
        for i, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch_index}"), start=1):
            input, label = data
            input = input.to(self.cfg.env.device)
            label = label.to(self.cfg.env.device)
            optimizer.zero_grad()
            output = model(input)
            outputs.append(output)
            labels.append(label)
            loss = self.loss_fn(output, label)
            loss.backward()
            optimizer.step()
```
在此使用，需要能够根据模型的输出和label进行计算

### 图像变换

```python
class Trainer:
    def __init__(self, cfg: Config):
        self.cfg: Config = cfg
        self.loss_fn = instantiate(cfg.train.loss_function)
        self.cur_fold = 1
        self.loss, self.f1_score, self.accuracy, self.confusion_matrix = 0, 0, 0, torch.zeros(
            (self.cfg.train.num_classes, self.cfg.train.num_classes), dtype=torch.int, device=self.cfg.env.device)
        self.train_transform = instantiate(self.cfg.train_transform)
        self.valid_transform = instantiate(self.cfg.valid_transform)
```
图像变化在此实例化
```python
        for train_ds, valid_ds in instantiate(self.cfg.dataset, data_folder_path=self.cfg.env.data_folder_path,
                                              train_transform=self.train_transform,
                                              valid_transform=self.valid_transform):
 
```
在此使用，因此需要在数据集中进行相应的处理。

## 数据处理相关类

本框架基于csv来管理图像。对于官方提供的数据集，使用`OfficialClaDataOrganizer.py`来
将图像集中在同一文件夹下，并生成`ground_truth.csv`，一列是
file_name，存储的是各图像的文件名，带后缀名，另一列是label，是各图像对应的标签。

图像增广后的图像也是用csv进行管理，例如经过Rotate变换后的图像都存放在同一文件夹中，并配有对应的
`ground_truth.csv`

### TableDataset

正对乳腺影像赛题数据集，专门设计了`TableDataset`类。该类接收一个pd.DataFrame，一列为file_name，
即图像名称，带后缀名。第二列是label，是图像对应的标签。

### 图像增广

`dataAugmentation.py`负责图像增广，并生成对应的`ground_truth.csv`

### `ClaDataset.py`

核心是`make_table`函数，用处是将训练所需要的图像所在文件夹中的`ground_truth.csv`集中起来，
传给`TableDataset``

`getClaTrainValidDataset`和`ClaCrossValidation`分别用于单折和多折交叉验证所需的数据集。
目前主要正对分类任务进行设计，若要扩展到特征识别任务，只需更改`make_table`函数即可，增加特征识别的数据集对应选项。

`getClaTrainValidDataset`和`ClaCrossValidation`还支持图像增广数据，需要传入增广数据所在文件夹的list。
`splitAugmentedImage`会根据包含在验证集中的文件集的文件名来筛选出可以加入训练集的增广后的图像，避免数据泄露。

***这里的验证集划分比例都没有考虑增广后的数据，所以若引入很多增广后的数据，验证集的比例会远小于1/5***

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
