import torch
import torchvision
import torchvision.transforms as transforms
from torcheval.metrics.functional import multiclass_precision,multiclass_f1_score, multiclass_accuracy, multiclass_confusion_matrix
import tqdm
import os
import json
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from torch.utils.data import Dataset
from datetime import datetime
from TDSNet.TDSNet import TDSNet
from models.model4compare.GoogleNet import GoogleNet
from models.model4compare.AlexNet import AlexNet
from models.model4compare.VGG import VGG
from models.model4compare.NiN import NiN
from models.UnetClassifer.unet import PretrainedClassifier,UnetClassifier
from utils.datasetCheck import checkDataset # 数据集检查
from utils.earlyStopping import EarlyStopping # 提前停止
from utils.multiMessageFilter import MultiMessageFilter  # ! 把MultiMessageFilter放入/utils.multiMessageFilter.py文件中
from utils.tools import getDevice, create_transforms # getDevice获取设备，create_transforms根据json配置创建transforms对象
from utils.tools import save_checkpoint, load_checkpoint  # 保存和加载检查点
from utils.MyBlock.MyCrop import MyCrop
from utils.PILResize import PILResize
from utils.ClaDataset import getClaTrainValidData

MultiMessageFilter().setup()
# 配置
cfg = {
    "model": "UnetClassifier", # 模型选择
    "data": "BreastOriginal", # 数据集选择
    "epoch_num": 1000, # 训练的 epoch 数量
    "num_workers": 2, # 数据加载器的工作进程数量,注意此处太大会导致内存溢出，很容易无法训练
    "batch_size": 16, # 批处理大小
    "in_channels": 3, # 输入通道数（图像）
    "device": getDevice(), # 设备，自动检测无需修改
    "pin_memory": True if getDevice() == "cuda" else False, # 是否使用 pin_memory，无需修改
    "infoShowFrequency": 100, # 信息显示频率(每多少个 batch 输出一次信息)
    # 加入断点续训的配置
    "resume": False,  # 是否从检查点恢复训练
    "checkpoint_path": None, # 检查点路径，如果为空，则自动寻找最新的检查点
    "dataset_root": '/Users/huiyangzheng/Desktop/Project/Competition/GCAIAEC2024/AIC/TDS-Net/data/乳腺分类训练数据集/train_split', # 数据集根目录
    "debug": {
        "num_samples_to_show": 4, # 显示样本个数
    } ,
    "early_stopping": {
        "patience": 20, # 耐心值
        "min_delta": 0.001 # 最小变化
    },
    "dataset_root": {
        "train_dir": "/Users/huiyangzheng/Desktop/Project/Competition/GCAIAEC2024/AIC/TDS-Net/data/乳腺分类训练数据集/train_split/train_split_train",
        "test_dir": "/Users/huiyangzheng/Desktop/Project/Competition/GCAIAEC2024/AIC/TDS-Net/data/乳腺分类训练数据集/train_split/train_split_test",
        "split_ratio": 0.9
    }
    
}

# 模型配置，不同模型配置不同
model_cfg = {
    "PretrainedClassifier": {
        "backbone": "resnet50",
        "lr": 0.001,
        "pretrained": True,
        "loss_fn": "CrossEntropyLoss"
    },
    "UnetClassifier": {
        "backbone": "resnet50",
        "freeze_backbone": True,
        "lr": 0.001,
        "optimizer": "adam",
        "pretrained": True,
        "loss_fn": "CrossEntropyLoss"
    },
    "TDSNet": {
        "lr": 0.001,
        "batch_size": 4,
        "pretrained": True,
        "loss_fn": "CrossEntropyLoss"
    },
    "AlexNet": {
        "lr": 0.001,
        "batch_size": 4,
        "pretrained": True,
        "loss_fn": "CrossEntropyLoss"
    },
    "GoogleNet": {
        "lr": 0.001,
        "batch_size": 4,
        "pretrained": True,
        "loss_fn": "CrossEntropyLoss"
    },
    "VGG": {
        "lr": 0.001,
        "batch_size": 4,
        "pretrained": True,
        "loss_fn": "CrossEntropyLoss"
    },
    "NiN": {
        "lr": 0.001,
        "batch_size": 4,
        "pretrained": True,
        "loss_fn": "CrossEntropyLoss"
    }
}
transforms_cfg = {
    "transform_train": {
        # "transforms的方法名字": {"参数名1": 参数值1, "参数名2": 参数值2, ...}
        "MyCrop": {},  # 自定义裁剪（fx）
        "Resize": {"size": (400, 400)},
        "RandomHorizontalFlip": {},
        "RandomVerticalFlip": {},
        "RandomRotation": {"degrees": 30},
        "RandomAffine": {"degrees": 0, "translate": [0.1, 0.1]},
        "RandomPerspective": {"distortion_scale": 0.5, "p": 0.5},
        "ToTensor": {},
        "Normalize": {
            "mean": [0.4914, 0.4822, 0.4465],
            "std": [0.2023, 0.1994, 0.2010],
        },
    },
    "transform_test": {
        # "Resize": {"size": (400,400)},
        "MyCrop": {},
        "PILResize": {"size": (128,128)},
        "ToTensor": {},
        "Normalize": {
            "mean": [0.4914, 0.4822, 0.4465],
            "std": [0.2023, 0.1994, 0.2010],
        },
    },
}

custom_transforms = {
    'MyCrop': MyCrop,
    'PILResize': PILResize
    }

class OriginalTransformImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        自定义数据集类
        :param root_dir: 数据集根目录，包含多个类别文件夹
        :param transform: 预处理变换
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(entry.name for entry in os.scandir(root_dir) if entry.is_dir())
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.image_paths = []
        self.labels = []
        
        for cls in self.classes:
            images_dir = os.path.join(root_dir, cls, 'images')
            if not os.path.isdir(images_dir):
                logger.warning(f"类别 {cls} 下没有找到 images 目录，跳过")
                continue
            for root, _, fnames in sorted(os.walk(images_dir)):
                for fname in sorted(fnames):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                        path = os.path.join(root, fname)
                        self.image_paths.append(path)
                        self.labels.append(self.class_to_idx[cls])
        
        assert len(self.image_paths) == len(self.labels), "图像路径和标签数量不匹配"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


class OriginalImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        self.class_to_idx = {}
        self.classes = []
        map_class_to_idx = {
            "2类":1,
            "3类":2,
            "4A类":3,
            "4B类":4,
            "4C类":5,
            "5类":6,
        }
        # 获取所有类别（文件夹名称）
        class_names = sorted(entry.name for entry in os.scandir(root_dir) if entry.is_dir())
        for idx, class_name in enumerate(class_names):
            self.classes.append(class_name)
            self.class_to_idx[class_name] = idx

        # 遍历每个类别文件夹
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            images_dir = os.path.join(class_dir, 'images')
            if not os.path.isdir(images_dir):
                continue
            # 遍历 images 子文件夹中的所有图像文件
            for root, _, fnames in sorted(os.walk(images_dir)):
                for fname in sorted(fnames):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                        path = os.path.join(root, fname)
                        item = (path, self.class_to_idx[class_name])
                        self.samples.append(item)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, target
    
    
def train_one_epoch(model, train_loader, epoch_index, num_class, tb_writer):
    '''
    训练一个 epoch
    :param model: 模型
    :param epoch_index: 当前 epoch
    :param train_loader: 训练数据加载器
    :param num_class: 类别数量
    :param tb_writer: TensorBoard 写入器
    '''
    model.train()
    running_loss = 0.0
    total_samples = 0

    all_outputs = []
    all_labels = []

    for i, data in enumerate(tqdm.tqdm(train_loader, desc=f"Epoch {epoch_index + 1}")):
        inputs, labels = data
        if cfg["device"] != "cpu":
            inputs = inputs.to(torch.device(cfg["device"]))
            labels = labels.to(torch.device(cfg["device"]))

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        # 累加损失和样本数量
        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

        # 收集所有的输出和标签
        all_outputs.append(outputs)
        all_labels.append(labels)

        # 可选：记录中间损失到 TensorBoard（保留原有功能）
        frequency = cfg["infoShowFrequency"]
        is_last_batch = (i == len(train_loader) - 1)
        if (i % frequency == frequency - 1) or is_last_batch:
            # 计算到目前为止的平均损失
            avg_loss = running_loss / total_samples
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar('Loss/train', avg_loss, tb_x)

    # 在整个 epoch 后，计算指标
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)

    avg_loss = running_loss / total_samples
    accuracy = multiclass_accuracy(all_outputs, all_labels).item()
    precision = multiclass_precision(all_outputs, all_labels, average='macro', num_classes=num_class).item()
    f1 = multiclass_f1_score(all_outputs, all_labels, average='macro', num_classes=num_class).item()

    return avg_loss, precision, f1, accuracy


def modelSelector(model_name, lr, num_class):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if model_name == 'TDSNet':
        model = TDSNet(num_class)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        return model, SummaryWriter('runs/TDS_' + str(lr) + "_" + timestamp), optimizer, timestamp
    elif model_name == 'AlexNet':
        model = AlexNet(num_class)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        return model, SummaryWriter('runs/AlexNet_' + str(lr) + "_" + timestamp), optimizer, timestamp
    elif model_name == 'GoogleNet':
        model = GoogleNet(num_class)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        return model, SummaryWriter('runs/GoogleNet_' + str(lr) + "_" + timestamp), optimizer, timestamp
    elif model_name == 'VGG':
        model = VGG(((1, 64), (1, 128), (2, 256), (2, 512), (2, 512)), num_class)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        return model, SummaryWriter('runs/VGG_' + str(lr) + "_" + timestamp), optimizer, timestamp
    elif model_name == 'NiN':
        model = NiN(num_class)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        return model, SummaryWriter('runs/NiN_' + str(lr) + "_" + timestamp), optimizer, timestamp
    elif model_name == 'PretrainedClassifier':
        model = PretrainedClassifier(num_classes=num_class, in_channels=cfg['in_channels'],
                                     backbone=model_cfg[model_name]["backbone"],
                                     pretrained=model_cfg[model_name]["pretrained"])
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        # 加载模型参数
        # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        return model, SummaryWriter('runs/Unet_' + str(lr) + "_" + timestamp), optimizer, timestamp
    elif model_name == 'UnetClassifier':
        model = UnetClassifier(num_classes=num_class, in_channels=cfg['in_channels'],
                               backbone=model_cfg[model_name]["backbone"],
                               pretrained=model_cfg[model_name]["pretrained"])
        if model_cfg[model_name]["freeze_backbone"]:
            model.freeze_backbone()
            print("freeze_backbone")
        if model_cfg[model_name]["optimizer"] == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif model_cfg[model_name]["optimizer"] == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        return model, SummaryWriter('runs/Unet_' + str(lr) + "_" + timestamp), optimizer, timestamp


def dataSelector(data='Breast'):
    """
    """
    if data == "BreastOriginalTransform":
        # 假设数据集根目录为当前工作目录下的 'dataset'
        dataset_root = cfg["dataset_root"]
        if not os.path.isdir(dataset_root):
            raise FileNotFoundError(f"数据集根目录 {dataset_root} 不存在")
        
        # 读取transform_cfg, 创建 transform_train 和 transform_test
        transform_train = create_transforms(transforms_cfg["transform_train"], custom_transforms=custom_transforms)
        transform_test = create_transforms(transforms_cfg["transform_test"], custom_transforms=custom_transforms)
        print("transform_train: ", transform_train)
        print("transform_test: ", transform_test)

        # 创建自定义数据集
        # full_dataset = CustomImageDataset(root_dir=dataset_root, transform=transform_train)
        train_ds_path = os.path.join(dataset_root, "train")
        train_ds = OriginalImageDataset(root_dir=dataset_root, transform=transform_train)
        valid_ds = OriginalImageDataset(root_dir=dataset_root, transform=transform_test)

        # 划分训练集和验证集（例如 80% 训练，20% 验证）
        total_size = len(full_dataset)
        valid_size = int(0.2 * total_size)
        train_size = total_size - valid_size
        train_ds, valid_ds = torch.utils.data.random_split(full_dataset, [train_size, valid_size],
                                                           generator=torch.Generator().manual_seed(42))

        # 为验证集设置不同的 transform
        valid_ds.dataset.transform = transform_test  # 修改验证集的 transform

    if data == "Breast":
        dest_dir = os.path.join(os.getcwd(), "data", "breast", "train_valid_test")
        train_dir = os.path.join(os.getcwd(), "data", "breast", "train", "cla")
        test_dir = os.path.join(os.getcwd(), "data", "breast", "test_A", "cla")
        # 读取 transform_cfg，创建 transform_train 和 transform_test
        # transform_train = create_transforms(transforms_cfg["transform_train"])
        # transform_test = create_transforms(transforms_cfg["transform_test"])
        transform_train = create_transforms(transforms_cfg["transform_train"], custom_transforms=custom_transforms)
        transform_test = create_transforms(transforms_cfg["transform_test"], custom_transforms=custom_transforms)
        print("transform_train: ", transform_train)
        print("transform_test: ", transform_test)

        # 创建数据集
        train_ds, train_valid_ds = [
            torchvision.datasets.ImageFolder(
                os.path.join(dest_dir, folder),
                transform=transform_train) for folder in ['train', 'train_valid']
        ]
        valid_ds, test_ds = [
            torchvision.datasets.ImageFolder(
                os.path.join(dest_dir, folder),
                transform=transform_test) for folder in ['valid', 'test']
        ]

    if data == "BreastOriginal":
        # 读取 transform_cfg，创建 transform_train 和 transform_test
        transform_train = create_transforms(transforms_cfg["transform_train"], custom_transforms=custom_transforms)
        transform_test = create_transforms(transforms_cfg["transform_test"], custom_transforms=custom_transforms)
        print("transform_train: ", transform_train)
        print("transform_test: ", transform_test)

        train_ds = OriginalImageDataset(root_dir=cfg["dataset_root"]["train_dir"], transform=transform_train)
        valid_ds = OriginalImageDataset(root_dir=cfg["dataset_root"]["test_dir"], transform=transform_test)

    elif data == 'FashionMNIST':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize((256, 256)),
             transforms.Normalize((0.5,), (0.5,))])
        train_ds = torchvision.datasets.FashionMNIST('./data', train=True, transform=transform, download=True)
        valid_ds = torchvision.datasets.FashionMNIST('./data', train=False, transform=transform, download=True)
    return train_ds, valid_ds


if __name__ == '__main__':
    "----------------------------------- data ---------------------------------------------"
    print("-------------------------- preparing data... --------------------------")
    train_ds, valid_ds = dataSelector(cfg["data"])  # 数据集选择fg["data"])  # 数据集选择
    num_class = len(train_ds.classes)  # 获取类别数量

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                                               pin_memory=cfg["pin_memory"], drop_last=True,
                                               num_workers=cfg["num_workers"])
    valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=cfg["batch_size"], shuffle=False,
                                               pin_memory=cfg["pin_memory"], drop_last=True,
                                               num_workers=cfg["num_workers"])
    # 检查数据集，输出相关信息
    checkDataset(train_ds, valid_ds, train_loader, valid_loader, cfg["debug"]["num_samples_to_show"])
    "----------------------------------- loss function ---------------------------------------------"
    # if model_cfg[cfg["model"]]["loss_fn"] == "CrossEntropyLoss":
    loss_fn = torch.nn.CrossEntropyLoss()

    "----------------------------------- model ---------------------------------------------"
    print("-------------------------- preparing model... --------------------------")
    model, writer, optimizer, timestamp = modelSelector(cfg["model"], model_cfg[cfg["model"]]["lr"], num_class)
    model.to(torch.device(cfg["device"]))
    print(f"cfg: {cfg}")
    print(f"model_cfg: {model_cfg[cfg['model']]}")


    "----------------------------------- resume from checkpoint ---------------------------------------------"
    epoch_number = 0
    best_vloss = float('inf')

    # 定义检查点路径
    checkPoint_path = os.path.join(os.getcwd(), 'checkPoint', f'{cfg["model"]}_{cfg["data"]}_{timestamp}')
    if not os.path.exists(checkPoint_path):
        os.makedirs(checkPoint_path)

    # 将 cfg 和 model_cfg 保存到 json 文件（仅在第一次运行时保存）
    if not os.path.exists(os.path.join(checkPoint_path, 'cfg.json')):
        with open(os.path.join(checkPoint_path, 'cfg.json'), 'w') as f:
            json.dump(cfg, f)
    if not os.path.exists(os.path.join(checkPoint_path, 'model_cfg.json')):
        with open(os.path.join(checkPoint_path, 'model_cfg.json'), 'w') as f:
            json.dump(model_cfg[cfg['model']], f)
    if not os.path.exists(os.path.join(checkPoint_path, 'transforms_cfg.json')):
        with open(os.path.join(checkPoint_path, 'transforms_cfg.json'), 'w') as f:
            json.dump(transforms_cfg, f)

    if cfg["resume"]:
        if cfg["checkpoint_path"]:
            checkpoint_path = cfg["checkpoint_path"]
        else:
            # 自动寻找最新的检查点
            checkpoint_path = os.path.join(checkPoint_path, 'resume_checkpoint', 'checkpoint.pth.tar')
            checkpoint_path = os.path.join(checkPoint_path, 'resume_checkpoint', 'checkpoint.pth.tar')
            if not os.path.exists(checkpoint_path):
                # 如果当前目录下没有检查点，则尝试在 checkPoint 目录中寻找
                checkpoint_dir = os.path.join(os.getcwd(), 'checkPoint')
                all_subdirs = [os.path.join(checkpoint_dir, d) for d in os.listdir(checkpoint_dir)
                               if os.path.isdir(os.path.join(checkpoint_dir, d))]
                if all_subdirs:
                    latest_subdir = max(all_subdirs, key=os.path.getmtime)
                    checkpoint_path = os.path.join(latest_subdir, 'checkpoint.pth.tar')
                else:
                    checkpoint_path = None
        if checkpoint_path and os.path.exists(checkpoint_path):
            model, optimizer, epoch_number, best_vloss = load_checkpoint(model, optimizer, checkpoint_path)
        else:
            print("=> No checkpoint found, starting from scratch")
    else:
        print("=> Starting training from scratch")

    "----------------------------------- training ---------------------------------------------"
    # 实例化 EarlyStopping，设定耐心值和最小变化
    early_stopping = EarlyStopping(cfg["early_stopping"]["patience"], cfg["early_stopping"]["min_delta"])

    best_vloss = 1_000_000.
    print("-------------------------- start training... --------------------------")
    print(f'time: {datetime.now()}')
    print(f"device: {cfg['device']}")
    for epoch in range(cfg["epoch_num"]):

        # 确保梯度跟踪开启，并对数据进行一次遍历
        avg_loss, avg_precision, avg_f1, avg_accuracy = train_one_epoch(model, train_loader, epoch, num_class, writer)

        # 验证模型
        model.eval()
        running_vloss = 0.0
        total_vsamples = 0
        all_voutputs = []
        all_vlabels = []

        with torch.no_grad():
            for i, vdata in enumerate(valid_loader):
                vinputs, vlabels = vdata
                if cfg["device"] != "cpu":
                    vinputs = vinputs.to(torch.device(cfg["device"]))
                    vlabels = vlabels.to(torch.device(cfg["device"]))
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)

                # 累加损失和样本数量
                batch_size = vinputs.size(0)
                running_vloss += vloss.item() * batch_size
                total_vsamples += batch_size

                # 收集所有的输出和标签
                all_voutputs.append(voutputs)
                all_vlabels.append(vlabels)

        # 计算验证集的平均损失和指标
        avg_vloss = running_vloss / total_vsamples
        all_voutputs = torch.cat(all_voutputs)
        all_vlabels = torch.cat(all_vlabels)
        vaccuracy = multiclass_accuracy(all_voutputs, all_vlabels).item()
        vprecision = multiclass_precision(all_voutputs, all_vlabels, average='macro', num_classes=num_class).item()
        vf1 = multiclass_f1_score(all_voutputs, all_vlabels, average='macro', num_classes=num_class).item()

        print('LOSS      train {:.4f} valid {:.4f}'.format(avg_loss, avg_vloss))
        print('ACCURACY  train {:.4f} valid {:.4f}'.format(avg_accuracy, vaccuracy))
        print('PRECISION train {:.4f} valid {:.4f}'.format(avg_precision, vprecision))
        print('F1        train {:.4f} valid {:.4f}'.format(avg_f1, vf1))

        # 将训练和验证的损失与指标记录到 TensorBoard
        writer.add_scalars('Loss', {'Train': avg_loss, 'Validation': avg_vloss}, epoch + 1)
        writer.add_scalars('Accuracy', {'Train': avg_accuracy, 'Validation': vaccuracy}, epoch + 1)
        writer.add_scalars('Precision', {'Train': avg_precision, 'Validation': vprecision}, epoch + 1)
        writer.add_scalars('F1_Score', {'Train': avg_f1, 'Validation': vf1}, epoch + 1)
        writer.flush()

        # 检查早停条件
        early_stopping(avg_vloss)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break

        # 追踪最佳性能，并保存模型状态
        is_best = avg_vloss < best_vloss


        if is_best:
            best_vloss = avg_vloss
            print(f"=> Validation loss improved to {avg_vloss:.6f} - saving best model")
            modelCheckPoint_path = os.path.join(checkPoint_path, 'model')

            # 保存检查点（包括 epoch，model_state_dict，optimizer_state_dict，best_vloss，但仅在最佳时保存）
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_vloss': best_vloss
            }
            # if not os.path.exists(os.path.join(checkPoint_path, 'resume_checkpoint')):
            #     os.makedirs(os.path.join(checkPoint_path, 'resume_checkpoint'))
            # save_checkpoint(checkpoint, checkPoint_path, filename=f'resume_checkpoint/epoch{epoch + 1}_vloss{avg_vloss:.4f}_precision{avg_vprecision:.4f}_f1{avg_vf1:.4f}.pth.tar')
            
            # 保存最佳模型参数
            if not os.path.exists(modelCheckPoint_path):
                os.makedirs(modelCheckPoint_path)
            torch.save(model.state_dict(), os.path.join(modelCheckPoint_path, f'{cfg["model"]}_best.pth'))

        # 记录训练过程的性能变化
        with open(os.path.join(checkPoint_path, 'training_log.txt'), 'a') as f:
            f.write(f"Epoch {epoch + 1}: vloss={avg_vloss:.4f}, accuracy={vaccuracy:.4f}, precision={vprecision:.4f}, f1={vf1:.4f}\n")
        epoch_number += 1

    print("-------------------------- training finished --------------------------")
    print(f'time: {datetime.now()}')