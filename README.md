from hydra.core.config_store import ConfigStore

# 训练框架说明

## 概述

本训练框架主要由三部分组成：

1. hydra超参数配置
2. Trainer类
3. 数据处理相关类

## hydra超参数配置

### 配置树

hydra库中的配置是树状结构

```python
def init_config():
    """
    初始化配置
    :return:
    """
    # 初始化
    cs = ConfigStore.instance()

    cs.store(group='train', name="default", node=DefaultTrainConfig)
    cs.store(group='train', name="sanity_check", node=FashionMNISTTrainConfig)

    cs.store(group='model', name="default", node=DefaultModelConfig)
    cs.store(group='model', name="sanity_check", node=AlexNetModelConfig)

    cs.store(group='dataset', name="single", node=SingleFoldDatasetConfig)
    cs.store(group='dataset', name="multiple", node=CrossValidationDatasetConfig)
    cs.store(group='dataset', name="sanity_check", node=FashionMNISTDatasetConfig)

    cs.store(group='optimizer', name="default", node=DefaultOptimizerConfig)

    cs.store(group='env', name="fx", node=FXEnvConfig)
    cs.store(group='env', name="zhy", node=ZHYEnvConfig)
    cs.store(group='env', name="yzl", node=YZLEnvConfig)

    cs.store(group='train_transform', name="default", node=DefaultTrainTransformConfig)

    cs.store(group='valid_transform', name="default", node=DefaultValidTransformConfig)

    # 初始化
    cs.store(name="config", node=Config)
```
上面这段代码中就定义了一棵配置树，根结点的名称是config，它的孩子结点有train,model,dataset,optimizer,env,train_transform,valid_transform。这些孩子结点每个都是一个group，
例如env这个group下面，有fx,zhy,yzl这三种不同的配置。同一group内的不同配置能够十分方便地相互切换。

node参数将配置树中的结点与python中的类关联起来，这些类要求用`@dataclass`进行装饰

***要注意的是，上面的代码中，node=后跟的是python中的类的名字，而group,name传入的是字符串，而hydra库所接受到的信息只有那些字符串，与我们起的类名没有关系***

***只有使用cs.store进行定义的配置才能够使用，光写对应的类是没有用的***

当我们想在某个函数中使用配置的参数时，需要用`@hydra.main`进行装饰：

```python
import hydra
@hydra.main(version_base=None,config_name="config")
def foo(cfg:Config):
  pass
```
上面这段代码的关键在于`config_name="config"`，这用于引导hydra库使用以`name="config"`作为根节点的配置树（与类名无关）。`foo(cfg:Config)`出现的`Config`类是typing，帮助ide进行各种提示。

注意，根节点的名字不强制要求为config。

### 与根结点关联的类

```python
defaults = [
    {"train": "sanity_check"},
    {"model": "sanity_check"},
    {"dataset": "sanity_check"},
    {"optimizer": "default"},
    {"env": "fx"},
    {"train_transform": "default"},
    {"valid_transform": "default"}
]

@dataclass
class Config:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    train: Any = MISSING
    model: Any = MISSING
    dataset: Any = MISSING
    optimizer: Any = MISSING
    env: Any = MISSING
    train_transform: Any = MISSING
    valid_transform: Any = MISSING
```

上述代码定义了与根节点config关联的类Config，是根节点所特有的定义方法。defaults说明了config根节点下的各个group默认使用哪种配置，注意这里指定的是结点的名称，而非类的名称。

在命令行方式下运行时，可以通过参数来选择组内使用的配置

```commandline
python train.py env=zhy
```

上述命令传入参数env=zhy，覆盖了default中的`{"env":"fx"}`

### 与其他结点关联的类

```python
@dataclass
class ClaTrainConfig:
    num_classes:int = 6
```
这是一个最简单的与其他结点关联的类的样例。如果用`cs.store(group='train',name='clatrain',node=ClaTrainConfig)`来初始化的话，在程序中可以用`cfg.train.num_classes`来访问，前提是在`defaults`中设置
`{"train":"clatrain"}`，或者在命令行中对默认配置进行覆盖。

### 用配置来实例化对象

如果想用配置来实例化对象，即在配置中描述一个对象，然后在程序中实例化，可以使用`_target_`来进行描述：

```python
@dataclass
class AlexNetModelConfig:
    _target_:str = "models.model4compare.AlexNet.AlexNet"
    num_classes: int = 10
```
用`cs.store(group='model',name='sanity_check')`，然后修改defaults，在程序中可以用`instantiate(cfg.model)`来获取该对象。`_target_`是在AlexNetModelConfig类所在文件import
 AlexNet的路径，num_classes是实例化AlexNet所需要的参数。如果类中有'_target_'项，hydra库就会把其它的配置项都用于实例化该类，所以，不能缺少实例化所需的参数，如果必须包含额外的信息，例如dataset配置中，
想要包含num_classes信息，但是实例化时又用不到num_classes参数，可以在定义类时，使用`**kwargs`来接受额外的关键字参数，如果是第三方库中的类，可以自行包装一下。

如果实例化所需要的参数要在运行时才能获取，例如optimizer的params，需要运行时从model获取。则可以在定义配置时，用符合要求的数据暂时代替，在程序中调用`instantiate`函数时以关键字的形式传入。

### 配置继承

```python
@dataclass
class EnvConfig:
    device: str = getDevice()
    pin_memory: bool = getDevice() == "cuda"


@dataclass
class FXEnvConfig(EnvConfig):
    data_folder_path: Path = os.path.join(os.curdir, 'data')
```
可以用继承的方式来避免重复配置相同项。子类中的配置会覆盖父类中的相同配置项。父类也需要用`@dataclass`进行装饰

### 配置中的类型

配置中可以使用`str,int,float,Path`类，同时也支持自定义的类（要用field函数）和`list`：

```python
@dataclass
class DefaultTrainConfig(ClaTrainConfig):
    checkpoint_path: Path = ''
    epoch_num: int = 1000
    num_workers: int = 2
    batch_size: int = 16
    info_frequency: int = 100
    early_stopping: EarlyStopping = field(default_factory=EarlyStopping)
    loss_function: LossFunction = field(default_factory=LossFunction)
```

要注意的是，不支持`tuple`，并且，类型为`list`的是，最好指定`_convert_="all"`，来确保配置生成的参数是`list`，否则会是hydra库自定义的类。
```python
@dataclass
class ResizeConfig:
    """
    这里设置_convert_="all"是为了让size在传入参数是变为list类型，否则会以hydra库中的类传入，不符合规定
    注意，conver只支持转换为list，不支持转换为tuple
    """
    _target_: str = "torchvision.transforms.Resize"
    size: List[int] = field(default_factory=lambda: [256, 256])
    _convert_: str = "all"
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
