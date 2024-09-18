import torch
import torchvision
import torchvision.transforms as transforms
from torcheval.metrics.functional import multiclass_precision, multiclass_f1_score
import tqdm
import os
import json
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
# from TDSNet.TDSNet import TDSNet
from TDSNet.TDSNet import TDSNet
from models.model4compare.GoogleNet import GoogleNet
from models.model4compare.AlexNet import AlexNet
from models.model4compare.VGG import VGG
from models.model4compare.NiN import NiN

from models.UnetClassifer.unet import UnetClassifier
from utils.datasetCheck import checkDataset # 数据集检查
from utils.earlyStopping import EarlyStopping # 提前停止
from utils.multiMessageFilter import setup_custom_logger  #! 把MultiMessageFilter放入/utils.multiMessageFilter.py文件中
logger = setup_custom_logger() # 设置日志屏蔽器，屏蔽f1_score的warning
from utils.tools import getDevice, create_transforms # getDevice获取设备，create_transforms根据json配置创建transforms对象
from utils.tools import save_checkpoint, load_checkpoint  # 保存和加载检查点
from utils.MyBlock.MyCrop import MyCrop

# 配置
cfg = {
    "model": "Unet", # 模型选择
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
    "checkpoint_path": "/Users/huiyangzheng/Desktop/Project/Competition/GCAIAEC2024/AIC/TDS-Net/checkPoint/Unet_Breast_20240913_115446/resume_checkpoint/epoch1_vloss1.7540_precision0.3054_f10.2189.pth.tar",  # 检查点路径，如果为空，则自动寻找最新的检查点
    "debug": {
        "num_samples_to_show": 4, # 显示样本个数
    }
}

# 模型配置，不同模型配置不同
model_cfg = {
    "Unet": {
        "backbone": "resnet50",
        "lr": 0.001,
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

# transform_train = torchvision.transforms.Compose([
#     torchvision.transforms.Resize((40, 40)),  # 缩放到 40x40 像素
#     # torchvision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0), ratio=(1.0, 1.0)),  # 随机裁剪并缩放到 32x32
#     torchvision.transforms.RandomHorizontalFlip(),  # 随机水平翻转
#     torchvision.transforms.RandomVerticalFlip(),  # 随机垂直翻转
#     torchvision.transforms.RandomRotation(degrees=30),  # 随机旋转 -30 到 30 度
#     torchvision.transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移
#     torchvision.transforms.RandomPerspective(distortion_scale=0.5, p=0.5),  # 随机网格畸变
#     torchvision.transforms.ToTensor(),  # 转换为张量
#     torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])  # 归一化
#     ])

transforms_cfg = {
    "transform_train": {
        # "transforms的方法名字": {"参数名1": 参数值1, "参数名2": 参数值2, ...}
        "MyCrop": {}, # 自定义裁剪（fx）
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
        "Resize": {"size": (400,400)},
        "ToTensor": {},
        "Normalize": {
            "mean": [0.4914, 0.4822, 0.4465],
            "std": [0.2023, 0.1994, 0.2010],
        },
    },
}

custom_transforms = {'MyCrop': MyCrop}


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
        model_name = TDSNet(num_class)
        optimizer = torch.optim.SGD(model_name.parameters(), lr=lr, momentum=0.9)
        return model_name, SummaryWriter('runs/TDS_' + str(lr) + "_" + timestamp), optimizer, timestamp
    elif model_name == 'AlexNet':
        model_name = AlexNet(num_class)
        optimizer = torch.optim.SGD(model_name.parameters(), lr=lr, momentum=0.9)
        return model_name, SummaryWriter('runs/AlexNet_' + str(lr) + "_" + timestamp), optimizer, timestamp
    elif model_name == 'GoogleNet':
        model_name = GoogleNet(num_class)
        optimizer = torch.optim.SGD(model_name.parameters(), lr=lr, momentum=0.9)
        return model_name, SummaryWriter('runs/GoogleNet_' + str(lr) + "_" + timestamp), optimizer, timestamp
    elif model_name == 'VGG':
        model_name = VGG(((1, 64), (1, 128), (2, 256), (2, 512), (2, 512)), num_class)
        optimizer = torch.optim.SGD(model_name.parameters(), lr=lr, momentum=0.9)
        return model_name, SummaryWriter('runs/VGG_' + str(lr) + "_" + timestamp), optimizer, timestamp
    elif model_name == 'NiN':
        model_name = NiN(num_class)
        optimizer = torch.optim.SGD(model_name.parameters(), lr=lr, momentum=0.9)
        return model_name, SummaryWriter('runs/NiN_' + str(lr) + "_" + timestamp), optimizer, timestamp
    elif model_name == 'Unet':
        model_name = UnetClassifier(num_classes=num_class, in_channels=cfg['in_channels'],
                               backbone=model_cfg[model_name]["backbone"], pretrained=model_cfg[model_name]["pretrained"])
        optimizer = torch.optim.SGD(model_name.parameters(), lr=lr, momentum=0.9)
        # 加载模型参数
        # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        return model_name, SummaryWriter('runs/Unet_' + str(lr) + "_" + timestamp), optimizer, timestamp


def dataSelector(data='Breast'):
    """
    #todo:
    """
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
            for i, vdata in enumerate(train_loader):
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
                
    print("-------------------------- training finished --------------------------")
    print(f'time: {datetime.now()}')
