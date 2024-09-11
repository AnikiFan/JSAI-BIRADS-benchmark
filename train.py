import torch
import torchvision
import torchvision.transforms as transforms
from torcheval.metrics.functional import multiclass_precision, multiclass_f1_score
import tqdm
import os
import os
import json
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
# from TDSNet.TDSNet import TDSNet
from TDSNet.TDSNet import TDSNet
from model4compare.GoogleNet import GoogleNet
from model4compare.AlexNet import AlexNet
from model4compare.VGG import VGG
from model4compare.NiN import NiN
import logging

from models.UnetClassifer.unet import UnetClassifier
from utils.datasetCheck import checkDataset # 数据集检查
from utils.earlyStopping import EarlyStopping # 提前停止
from utils.multiMessageFilter import setup_custom_logger  #! 把MultiMessageFilter放入/utils.multiMessageFilter.py文件中
logger = setup_custom_logger() # 设置日志屏蔽器，屏蔽f1_score的warning
from utils.tools import getDevice, create_transforms # getDevice获取设备，create_transforms根据json配置创建transforms对象


# 配置
cfg = {
    "model": "Unet", # 模型选择
    "data": "Breast", # 数据集选择
    "epoch_num": 1000, # 训练的 epoch 数量
    "batch_size": 4, # 批处理大小
    "in_channels": 3, # 输入通道数（图像）
    "device": getDevice(), # 设备，自动检测无需修改
    "pin_memory": True if getDevice() == "cuda" else False, # 是否使用 pin_memory，无需修改
    "infoShowFrequency": 100 # 信息显示频率(每多少个 batch 输出一次信息)
}

# 模型配置，不同模型配置不同
model_cfg = {
    "Unet": {
        "backbone": "vgg",
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
        "Resize": {"size": (40, 40)},
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
        "Resize": {"size": (40, 40)},
        "ToTensor": {},
        "Normalize": {
            "mean": [0.4914, 0.4822, 0.4465],
            "std": [0.2023, 0.1994, 0.2010],
        },
    },
}



def train_one_epoch(model,train_loader,epoch_index,num_class, tb_writer):
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
    
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    # 改为带进度条的循环
    for i, data in enumerate(tqdm.tqdm(train_loader, desc=f"Epoch {epoch_index + 1}")):
        # Every data instance is an input + label pair
        inputs, labels = data
        if cfg["device"] != "cpu":
            inputs = inputs.to(torch.device(cfg["device"]))
            labels = labels.to(torch.device(cfg["device"]))

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        running_precision += multiclass_precision(outputs, labels).tolist()
        running_f1 += multiclass_f1_score(outputs, labels, average='macro', num_classes=num_class).tolist()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        frequency = cfg["infoShowFrequency"]
        if i % frequency == frequency-1:
            last_loss = running_loss / frequency  # loss per batch
            last_precision = running_precision / frequency  # loss per batch
            last_f1 = running_f1 / frequency  # loss per batch
            # 下面输出会与tqdm矛盾，建议注释掉？
            # print('  batch {} loss     : {}'.format(i + 1, last_loss))
            # print('           precision: {}'.format(i + 1, last_precision))
            # print('           f1       : {}'.format(i + 1, last_f1))
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            tb_writer.add_scalar('Precision/train', last_precision, tb_x)
            tb_writer.add_scalar('F1/train', last_f1, tb_x)
            running_loss = 0.
            running_precision = 0.
            running_f1 = 0.

    return last_loss, last_precision, last_f1


def modelSelector(model, lr, num_class):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if model == 'TDSNet':
        model = TDSNet(num_class)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        return model, SummaryWriter('runs/TDS_' + str(lr) + "_" + timestamp), optimizer, timestamp
    elif model == 'AlexNet':
        model = AlexNet(num_class)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        return model, SummaryWriter('runs/AlexNet_' + str(lr) + "_" + timestamp), optimizer, timestamp
    elif model == 'GoogleNet':
        model = GoogleNet(num_class)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        return model, SummaryWriter('runs/GoogleNet_' + str(lr) + "_" + timestamp), optimizer, timestamp
    elif model == 'VGG':
        model = VGG(((1, 64), (1, 128), (2, 256), (2, 512), (2, 512)), num_class)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        return model, SummaryWriter('runs/VGG_' + str(lr) + "_" + timestamp), optimizer, timestamp
    elif model == 'NiN':
        model = NiN(num_class)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        return model,SummaryWriter('runs/NiN_'+str(lr)+"_"+timestamp),optimizer,timestamp
    elif model == 'Unet':
        model = UnetClassifier(num_classes=num_class,in_channels = cfg['in_channels'], backbone=model_cfg[model]["backbone"],pretrained=model_cfg[model]["pretrained"])
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        return model,SummaryWriter('runs/Unet_'+str(lr)+"_"+timestamp),optimizer,timestamp


def dataSelector(data='Breast'):
    """
    #todo:
    """
    if data == "Breast":
        dest_dir = os.path.join(os.getcwd(), "data", "breast", "train_valid_test")
        train_dir = os.path.join(os.getcwd(), "data", "breast", "train", "cla")
        test_dir = os.path.join(os.getcwd(), "data", "breast", "test_A", "cla")
        # 读取transform_cfg,创建transform_train和transform_test
        transform_train = create_transforms(transforms_cfg["transform_train"])
        transform_test = create_transforms(transforms_cfg["transform_test"])
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
    train_ds, valid_ds = dataSelector('Breast') # 数据集选择
    num_class = len(train_ds.classes) # 获取类别数量
    
    # 创建数据加载器    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, pin_memory=cfg["pin_memory"], drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=cfg["batch_size"], shuffle=False, pin_memory=cfg["pin_memory"], drop_last=True)
    # 检查数据集，输出相关信息
    checkDataset(train_ds, valid_ds, train_loader, valid_loader,num_samples_to_show=0) # 
    "----------------------------------- loss function ---------------------------------------------"
    # if model_cfg[cfg["model"]]["loss_fn"] == "CrossEntropyLoss":
    loss_fn = torch.nn.CrossEntropyLoss()

    "----------------------------------- model ---------------------------------------------"
    print("-------------------------- preparing model... --------------------------")
    model,writer,optimizer,timestamp = modelSelector(cfg["model"], model_cfg[cfg["model"]]["lr"], num_class)
    model.to(torch.device(cfg["device"]))

    "----------------------------------- training ---------------------------------------------"
    epoch_number = 0

    # EPOCHS = 5
    
    # 实例化 EarlyStopping，设定耐心值和最小变化
    early_stopping = EarlyStopping(patience=10, min_delta=0.01)

    best_vloss = 1_000_000.
    print("-------------------------- start training... --------------------------")
    print(f'time: {datetime.now()}')
    print(f"device: {cfg['device']}")
    print(f"cfg: {cfg}")
    print(f"model_cfg: {model_cfg}")
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
                vf1 = multiclass_f1_score(voutputs, vlabels, average='macro', num_classes=num_class).tolist() #note:设置成macro进而计算每个类别的f1值
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
        if avg_vloss < best_vloss:
            print(f"Validation loss improved from {best_vloss:.6f} to {avg_vloss:.6f} - saving model")
            best_vloss = avg_vloss
            
            # if not os.path.exists('checkPoint'):
            #     os.makedirs('checkPoint')
            # if not os.path.exists(f'checkPoint/{cfg["model"]}_{cfg["data"]}_{timestamp}'):
            #     os.makedirs(f'checkPoint/{cfg["model"]}_{timestamp}')
            
            #e.g checkPoint/Unet_Breast_20210929_123456
            checkPoint_path = os.path.join(os.getcwd(), 'checkPoint', f'{cfg["model"]}_{cfg["data"]}_{timestamp}')
            if not os.path.exists(checkPoint_path):
                os.makedirs(checkPoint_path)
            
            # 将cfg和model_cfg保存到json文件
            # e.g checkPoint/Unet_Breast_20210929_123456/cfg.json
            with open(os.path.join(checkPoint_path, 'cfg.json'), 'w') as f:
                json.dump(cfg, f)
            with open(os.path.join(checkPoint_path, 'model_cfg.json'), 'w') as f:
                json.dump(model_cfg, f)
            with open(os.path.join(checkPoint_path, 'transforms_cfg.json'), 'w') as f:
                json.dump(transforms_cfg, f)
            
            modelCheckPoint_path = os.path.join(checkPoint_path, 'model')
            if not os.path.exists(modelCheckPoint_path):
                os.makedirs(modelCheckPoint_path)
            torch.save(model.state_dict(), modelCheckPoint_path+f'/{cfg["model"]}_ac{avg_vprecision:.6f}_f1{avg_vf1:.6f}_{epoch}.pth')
            optimizerCheckPoint_path = os.path.join(checkPoint_path, 'optimizer')
            if not os.path.exists(optimizerCheckPoint_path):
                os.makedirs(optimizerCheckPoint_path)
            torch.save(optimizer.state_dict(), optimizerCheckPoint_path+f'/{cfg["model"]}_ac{avg_vprecision:.6f}_f1{avg_vf1:.6f}_{epoch}.pth')

        epoch_number += 1
        
    print("-------------------------- training finished --------------------------")
    print(f'time: {datetime.now()}')
