import torch
import torchvision
import torchvision.transforms as transforms
from torcheval.metrics.functional import multiclass_precision, multiclass_f1_score
import tqdm
import os
import json
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from torch.utils.data import Dataset
from datetime import datetime
# from TDSNet.TDSNet import TDSNet
# from TDSNet.TDSNet import TDSNet
from models.model4compare.GoogleNet import GoogleNet
from models.model4compare.AlexNet import AlexNet
from models.model4compare.VGG import VGG
from models.model4compare.NiN import NiN

from models.UnetClassifer.unet import PretrainedClassifier,UnetClassifier
from utils.datasetCheck import checkDataset # 数据集检查
from utils.earlyStopping import EarlyStopping # 提前停止
from utils.multiMessageFilter import setup_custom_logger  #! 把MultiMessageFilter放入/utils.multiMessageFilter.py文件中
logger = setup_custom_logger() # 设置日志屏蔽器，屏蔽f1_score的warning
from utils.tools import getDevice, create_transforms # getDevice获取设备，create_transforms根据json配置创建transforms对象
from utils.tools import save_checkpoint, load_checkpoint  # 保存和加载检查点
from utils.MyBlock.MyCrop import MyCrop
from utils.PILResize import PILResize

# 配置
cfg = {
    "model": "UnetClassifier", # 模型选择
    "data": "Breast", # 数据集选择
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
        # "Resize": {"size": (400, 400)},
        "MyCrop": {}, 
        "PILResize": {"size": (128, 128)},
        # "RandomHorizontalFlip": {},
        # "RandomVerticalFlip": {},
        # "RandomRotation": {"degrees": 30},
        # "RandomAffine": {"degrees": 0, "translate": [0.1, 0.1]},
        # "RandomPerspective": {"distortion_scale": 0.5, "p": 0.5},
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

class OriginalImageDataset(Dataset):
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



def train_one_epoch(model, train_loader, epoch_index, num_class, tb_writer):
    '''
    训练一个 epoch
    :param model: 模型
    :param epoch_index: 当前 epoch
    :param train_loader: 训练数据加载器
    :param num_class: 类别数量
    :param tb_writer: TensorBoard 写入器
    '''
    running_loss = 0.
    running_precision = 0.
    running_f1 = 0.
    last_loss = 0.
    last_precision = 0.
    last_f1 = 0.

    model.train(True)

    for i, data in enumerate(tqdm.tqdm(train_loader, desc=f"Epoch {epoch_index + 1}")):
        inputs, labels = data
        if cfg["device"] != "cpu":
            inputs = inputs.to(torch.device(cfg["device"]))
            labels = labels.to(torch.device(cfg["device"]))

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()

        running_precision += multiclass_precision(outputs, labels).tolist()
        running_f1 += multiclass_f1_score(outputs, labels, average='macro', num_classes=num_class).tolist()

        optimizer.step()

        running_loss += loss.item()
        frequency = cfg["infoShowFrequency"]
        is_last_batch = (i == len(train_loader) - 1)
        if (i % frequency == frequency - 1) or is_last_batch:
            # 计算实际的批次数
            batch_count = frequency if not is_last_batch else (i % frequency + 1)
            last_loss = running_loss / batch_count
            last_precision = running_precision / batch_count
            last_f1 = running_f1 / batch_count

            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            tb_writer.add_scalar('Precision/train', last_precision, tb_x)
            tb_writer.add_scalar('F1/train', last_f1, tb_x)

            running_loss = 0.
            running_precision = 0.
            running_f1 = 0.

    return last_loss, last_precision, last_f1


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
                               backbone=model_cfg[model_name]["backbone"], pretrained=model_cfg[model_name]["pretrained"])
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        # 加载模型参数
        # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        return model, SummaryWriter('runs/Unet_' + str(lr) + "_" + timestamp), optimizer, timestamp
    elif model_name == 'UnetClassifier':
        model = UnetClassifier(num_classes=num_class, in_channels=cfg['in_channels'],
                               backbone=model_cfg[model_name]["backbone"], pretrained=model_cfg[model_name]["pretrained"])
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
    if data == "BreastOriginal":
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
        # 读取transform_cfg,创建transform_train和transform_test
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
    train_ds, valid_ds = dataSelector('Breast')  # 数据集选择
    num_class = len(train_ds.classes)  # 获取类别数量

    # 创建数据加载器    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, pin_memory=cfg["pin_memory"], drop_last=True, num_workers=cfg["num_workers"])
    valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=cfg["batch_size"], shuffle=False, pin_memory=cfg["pin_memory"], drop_last=True, num_workers=cfg["num_workers"])
    # 检查数据集，输出相关信息
    checkDataset(train_ds, valid_ds, train_loader, valid_loader, cfg["debug"]["num_samples_to_show"])  #
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

    # 将cfg和model_cfg保存到json文件（仅在第一次运行时保存）
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
            checkpoint_path = os.path.join(checkPoint_path,'resume_checkpoint' ,'checkpoint.pth.tar')
            if not os.path.exists(checkpoint_path):
                # 如果当前目录下没有检查点，则尝试在checkPoint目录中寻找
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
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)

    best_vloss = 1_000_000.
    print("-------------------------- start training... --------------------------")
    print(f'time: {datetime.now()}')
    print(f"device: {cfg['device']}")
    for epoch in range(cfg["epoch_num"]):

        # Make sure gradient tracking is on, and do a pass over the data
        avg_loss, avg_precision, avg_f1 = train_one_epoch(model, train_loader, epoch, num_class, writer)

        running_vloss = 0.0
        running_vprecison = 0.0
        running_vf1 = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(valid_loader):
                vinputs, vlabels = vdata
                if cfg["device"] != "cpu":
                    vinputs = vinputs.to(torch.device(cfg["device"]))
                    vlabels = vlabels.to(torch.device(cfg["device"]))
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                vprecision = multiclass_precision(voutputs, vlabels).tolist()
                vf1 = multiclass_f1_score(voutputs, vlabels, average='macro',
                                          num_classes=num_class).tolist()  # note:设置成macro进而计算每个类别的f1值
                running_vloss += vloss
                running_vprecison += vprecision
                running_vf1 += vf1

        avg_vloss = running_vloss / (i + 1)
        avg_vprecision = running_vprecison / (i + 1)
        avg_vf1 = running_vf1 / (i + 1)
        print('LOSS      train {} valid {}'.format(avg_loss, avg_vloss))
        print('PRECISION train {} valid {}'.format(avg_precision, avg_vprecision))
        print('F1        train {} valid {}'.format(avg_f1, avg_vf1))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': avg_loss, 'Validation': avg_vloss},
                           epoch_number + 1)
        writer.add_scalars('Training vs. Validation Precision',
                           {'Training': avg_precision, 'Validation': avg_vprecision},
                           epoch_number + 1)
        writer.add_scalars('Training vs. Validation F1',
                           {'Training': avg_f1, 'Validation': avg_vf1},
                           epoch_number + 1)
        writer.flush()

        # 检查早停条件
        early_stopping(avg_vloss)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break

        # Track best performance, and save the model's state
        is_best = avg_vloss < best_vloss
        
        if is_best:
            best_vloss = avg_vloss
            print(f"=> Validation loss improved to {avg_vloss:.6f} - saving best model")
            modelCheckPoint_path = os.path.join(checkPoint_path, 'model')
            
            # 保存checkpoint（包括epoch，model_state_dict，optimizer_state_dict，best_vloss，但仅在best时保存）
            # 保存断点重训所需的信息（需要包括epoch，model_state_dict，optimizer_state_dict，best_vloss）
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_vloss': best_vloss
            }
            if not os.path.exists(os.path.join(checkPoint_path, 'resume_checkpoint')):
                os.makedirs(os.path.join(checkPoint_path, 'resume_checkpoint'))
            save_checkpoint(checkpoint, checkPoint_path, filename=f'resume_checkpoint/epoch{epoch + 1}_vloss{avg_vloss:.4f}_precision{avg_vprecision:.4f}_f1{avg_vf1:.4f}.pth.tar')
            
            # 保存最佳模型参数
            if not os.path.exists(modelCheckPoint_path):
                os.makedirs(modelCheckPoint_path)
            torch.save(model.state_dict(), os.path.join(modelCheckPoint_path, f'{cfg["model"]}_best.pth'))
        
        # 记录训练过程的性能变化
        with open(os.path.join(checkPoint_path, 'training_log.txt'), 'a') as f:
            f.write(f"Epoch {epoch + 1}: vloss={avg_vloss:.4f}, precision={avg_vprecision:.4f}, f1={avg_vf1:.4f}\n")
        epoch_number += 1

    print("-------------------------- training finished --------------------------")
    print(f'time: {datetime.now()}')
