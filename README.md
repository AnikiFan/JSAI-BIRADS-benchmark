# 🤔 How to ... 

## 初始化框架

1. 按照仓库目录约定来整理`data`文件夹，只需将`official_test`,`official_trian`,`BUS`,`USG`按照约定放入即可，其中`BUS`和`USG`是事先处理过的版本（微信群里有）
2. 运行`\utils\OfficialClaDataOrganizer.py`,`\utils\OfficialFeaDataOrganizer.py`，来获取`train`和`test`数据集

note:cla和fea似乎需要都准备好才能进行数据增强？

## 数据增强

使用`\utils\dataAugmentations.py`。

增强后得到的数据自动存放在对应任务数据文件夹下的`augmented`文件夹下的文件夹中，文件夹的名称是所用变换的名称以及所用比例（fea任务不支持）。在该文件夹下，会有`README.md`文件详细描述所用变换的参数。

自动生成的文件夹名称可以自行修改。

当前图像增强的处理逻辑会检测文件夹是否会重名，如果会，则跳过。因此，如果第一次对cla任务进行Rotate增广，第二次也对cla任务进行Rotate增广，ratio不变，即使Rotate的参数变了，如果第一次得到的文件夹没有重命名，第二次的增广会自动跳过。

### MixUp

目前只支持cla任务，通过`official_train`，`BUS`和`USG`参数来指定应该使用哪些数据集进行增广

### Preprocess

支持cla任务和fea任务。

`transform`参数必须传入的是`A.Compose`，即使只有单个变换也许套上`A.Compose`

若为fea任务，将`fea_official_train`设置为`True`，将`official_train`，`BUS`和`USG`设为`False`。

若为cla任务，将`fea_official_train`设置为`False`，按需将`official_train`，`BUS`和`USG`设为`True`。

### 使用乳腺图像数据

`\utils\BreastDataset.py`提供了乳腺图像的单折数据集函数`getBreastTrainValidData`和多折交叉验证数据类`BreastCrossValidationData`。都是通过`next()`函数以迭代的方式获取所需训练集和验证集。

若为fea任务，将`fea_official_train`设置为`True`，将`official_train`，`BUS`和`USG`设为`False`。

若为cla任务，将`fea_official_train`设置为`False`，按需将`official_train`，`BUS`和`USG`设为`True`

### 添加配置

1. 在对应文件中编写配置类，例如想要配置FashionMNIST数据集配置，就在`\config\datasetConfig.py`中编写`FashionMNIST`类
    - 如果涉及到实例化，需要在target中指定导入路径
    - 要确保实例化的类符合框架要求

```
@dataclass
class FashionMNISTDatasetConfig:
    _target_: str = "data.FashionMNIST.MyFashionMNIST.MyFashionMNIST"
    num_classes: int = 10
```

2. 在`\config\config.py`中注册该配置

```
    cs.store(group='dataset', name="fashion_mnist", node=FashionMNISTDatasetConfig)
    cs.store(group='dataset', name="mnist", node=MNISTDatasetConfig)
    cs.store(group='dataset', name="cifar10", node=CIFAR10DatasetConfig)
 
```

### 使用`optuna`进行调参

在`\config\config.yaml`中的`hydra.sweeper`条目下对调参任务进行配置，重点是`params`项，详见`hydra`库文档中关于`optuna`的部分

运行调参任务时，需要附带`--multirun`参数，即在命令行中使用`python train.py --multirun`

### 使用`optuna-dashboard`对调参结果可视化

首先在`python`环境中安装sqlite库和`optuna-dashboard`，并在`vscode`中安装`optuna-dashboard`插件，右键调参任务生成的`db`文件，点击`Open in optuna dashboard`

# 仓库目录约定

## `config`

用于放置配置相关文件

## `data`

第一层将数据集分为不同类，例如`CIFAR10`,`FashionMNIST`,`breast`和用于DEBUG的`test`。

`breast`下进一步分为`fea`和`cla`。`cla`和`fea`下的`official_train`和`official_test`均为官方提供的训练集和测试集，没有做过任何改动。`train`和`test`为在此基础上整理过后得到的训练集和测试集。`augmented`存放数据增广得到的数据。

```
data
├── CIFAR10
│   └── __pycache__
├── FashionMNIST
│   ├── FashionMNIST
│   │   └── raw
│   ├── __pycache__
│   └── raw
├── MNIST
│   └── __pycache__
├── breast
│   ├── cla
│   │   ├── BUS
│   │   │   ├── Images
│   │   │   └── Masks
│   │   ├── OASBUD
│   │   ├── USG
│   │   ├── augmented
│   │   │   ├── ElasticTransform,ratio=(1.0,1.0,1.0,1.0,1.0,1.0)
│   │   │   ├── ElasticTransform,ratio=(2,1,3,4,5,6)
│   │   │   ├── Mixup,ratio=(1.0,1.0,1.0,1.0,1.0,1.0)
│   │   │   ├── Mixup,ratio=(2,1,3,4,5,6)
│   │   │   ├── Perspective,ratio=(1.0,1.0,1.0,1.0,1.0,1.0)
│   │   │   ├── Perspective,ratio=(2,1,3,4,5,6)
│   │   │   ├── RandomBrightnessContrast,ratio=(1.0,1.0,1.0,1.0,1.0,1.0)
│   │   │   ├── RandomBrightnessContrast,ratio=(2,1,3,4,5,6)
│   │   │   ├── Rotate,HorizontalFlip,ratio=(1.0,1.0,1.0,1.0,1.0,1.0)
│   │   │   ├── Rotate,HorizontalFlip,ratio=(2,1,3,4,5,6)
│   │   │   ├── Rotate,ratio=(1.0,1.0,1.0,1.0,1.0,1.0)
│   │   │   ├── Rotate,ratio=(2,1,3,4,5,6)
│   │   │   └── VerticalFlip,ratio=(2,1,3,4,5,6)
│   │   ├── official_test
│   │   │   ├── 2类
│   │   │   │   ├── images
│   │   │   │   └── labels
│   │   │   ├── 3类
│   │   │   │   ├── images
│   │   │   │   └── labels
│   │   │   ├── 4A类
│   │   │   │   ├── images
│   │   │   │   └── labels
│   │   │   ├── 4B类
│   │   │   │   ├── images
│   │   │   │   └── labels
│   │   │   ├── 4C类
│   │   │   │   ├── images
│   │   │   │   └── labels
│   │   │   └── 5类
│   │   │       ├── images
│   │   │       └── labels
│   │   ├── official_train
│   │   │   ├── 2类
│   │   │   │   ├── images
│   │   │   │   └── labels
│   │   │   ├── 3类
│   │   │   │   ├── images
│   │   │   │   └── labels
│   │   │   ├── 4A类
│   │   │   │   ├── images
│   │   │   │   └── labels
│   │   │   ├── 4B类
│   │   │   │   ├── images
│   │   │   │   └── labels
│   │   │   ├── 4C类
│   │   │   │   ├── images
│   │   │   │   └── labels
│   │   │   └── 5类
│   │   │       ├── images
│   │   │       └── labels
│   │   ├── test
│   │   └── train
│   └── fea
│       ├── augmented
│       │   └── Rotate
│       ├── official_test
│       │   ├── boundary_labels
│       │   ├── calcification_labels
│       │   ├── direction_labels
│       │   ├── images
│       │   └── shape_labels
│       ├── official_train
│       │   ├── boundary_labels
│       │   ├── calcification_labels
│       │   ├── direction_labels
│       │   ├── images
│       │   └── shape_labels
│       ├── test
│       └── train
└── test
```

## `docs`

存放`cla_order.csv`和`pre_order.csv`等文档

## `models`

存放模型

## `outputs`

`hydra`库生成日志的存放文件夹

## `runs`

`tensorboard`库生成日志的存放文件夹

## `test`

用于存放测试用的文件，`unitTest`为了避免`import`语句相关问题，暂时不放在此处

## `utils`

工具脚本

# 训练框架说明

## 概述

本训练框架主要由三部分组成：

1. hydra超参数配置:`\config`
2. Trainer类:`\utils\Trainer.py`
3. 数据处理相关类:`\utils\BreastDataset.py`,`TableDataset.py`,`\utils\OfficialClaDataOrganizer.py`,`\utils\OfficialFeaDataOrganizer.py`

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

***只有使用cs.store进行初始化的配置才能够使用，光写对应的类是没有用的***

```python
# Using the type
cs.store(name="config1", node=MySQLConfig)
# Using an instance, overriding some default values
cs.store(name="config2", node=MySQLConfig(host="test.db", port=3307))
```
第一种方式将config1与MySQLConfig类中定义的配置相关联。第二种方式则用`host="test.db"`和`port=3307`来进行覆盖，对学习率等参数进行微调的话第二种方式比较方便，
不用重复设置多个类。

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
 AlexNet的路径，num_classes是实例化AlexNet所需要的参数。如果类中有'_target_'项，hydra库就会把该类下面的其它的成员都作为用于实例化该类的参数，所以，该类的成员必须包含实例化所需的所有参数，如果必须包含额外的信息，例如dataset配置中，
想要包含num_classes信息，但是实例化时又用不到num_classes参数，可以在定义类时，使用`**kwargs`来接受额外的关键字参数，如果是第三方库中的类，可以自行包装一下。

如果实例化所需要的参数要在运行时才能获取，例如optimizer的params，需要运行时从model获取。则可以在定义配置时，用`omegaconf.MISSING`代替，在程序中调用`instantiate`函数时以关键字的形式传入。

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

配置中可以使用`str,int,float,Path`类，同时也支持自定义的类和`list`（要用`field`函数）：

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

要注意的是，不支持`tuple`，并且，类型为`list`时，最好指定`_convert_="all"`，来确保配置生成的参数是`list`，否则会是hydra库自定义的类。
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
数据集在此处被实例化，唯一的要求是数据集类是迭代器类型，每次迭代返回训练集和验证集，并且训练集和验证集能够用于实例化`DataLoader`类(因此要求数据集类中存储的是input,label对)。
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
在此使用，需要能够根据`model`的返回的`output`和`dataset`返回的`label`进行计算

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
传给`TableDataset`

`getClaTrainValidDataset`和`ClaCrossValidation`分别用于单折和多折交叉验证所需的数据集。
目前主要正对分类任务进行设计，若要扩展到特征识别任务，只需更改`make_table`函数即可，增加特征识别的数据集对应选项。

`getClaTrainValidDataset`和`ClaCrossValidation`还支持图像增广数据，需要传入由所用增广数据所在文件夹组成的list。
`splitAugmentedImage`会根据包含在验证集中的图像文件名来筛选出可以加入训练集的增广后的图像，避免数据泄露。

***这里的验证集划分比例都没有考虑增广后的数据，所以若引入很多增广后的数据，验证集的比例会远小于指定的比例***

# tensorboard使用方法

```commandline
tensorboard --logdir=./runs
```
