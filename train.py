import torch
import torchvision
import torchvision.transforms as transforms
from torcheval.metrics.functional import multiclass_precision, multiclass_f1_score
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from TDSNet import TDSNet
from model4compare.GoogleNet import GoogleNet
from model4compare.AlexNet import AlexNet
from model4compare.VGG import VGG
from model4compare.NiN import NiN
import logging


# 自定义过滤器类，用于过滤指定的日志消息
class MultiMessageFilter(logging.Filter):
    def __init__(self):
        super().__init__()
        self.filtered_messages = [
            "Warning: Some classes do not exist in the target. F1 scores for these classes will be cast to zeros.",
            "STREAM b'IHDR'",
            "STREAM b'IDAT'"
        ]

    def filter(self, record):
        # 如果消息与任意一个过滤消息匹配，则过滤掉
        return not any(msg in record.getMessage() for msg in self.filtered_messages)


# 创建 logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# 创建控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# 添加自定义过滤器
multi_message_filter = MultiMessageFilter()
console_handler.addFilter(multi_message_filter)

# 将处理器添加到 logger
logger.addHandler(console_handler)


def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    running_precision = 0.
    running_f1 = 0.
    last_loss = 0.
    last_precision = 0.
    last_f1 = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        if torch.cuda.is_available():
            inputs = inputs.to(torch.device('cuda'))
            labels = labels.to(torch.device('cuda'))

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
        if i % frequency == frequency-1:
            last_loss = running_loss / frequency  # loss per batch
            last_precision = running_precision / frequency  # loss per batch
            last_f1 = running_f1 / frequency  # loss per batch
            print('  batch {} loss     : {}'.format(i + 1, last_loss))
            print('  batch {} precision: {}'.format(i + 1, last_precision))
            print('  batch {} f1       : {}'.format(i + 1, last_f1))
            tb_x = epoch_index * len(training_loader) + i + 1
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
        return model, SummaryWriter('runs/NiN_' + str(lr) + "_" + timestamp), optimizer, timestamp


def dataSelector(data='Breast'):
    if data == "Breast":
        dest_dir = os.path.join(os.getcwd(), "data", "breast", "train_valid_test")
        train_dir = os.path.join(os.getcwd(), "data", "breast", "train", "cla")
        test_dir = os.path.join(os.getcwd(), "data", "breast", "test_A", "cla")
        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.Resize((256, 256)),  # 缩放到 40x40 像素
            # torchvision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0), ratio=(1.0, 1.0)),  # 随机裁剪并缩放到 32x32
            torchvision.transforms.RandomHorizontalFlip(),  # 随机水平翻转
            torchvision.transforms.RandomVerticalFlip(),  # 随机垂直翻转
            torchvision.transforms.RandomRotation(degrees=30),  # 随机旋转 -30 到 30 度
            torchvision.transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移
            torchvision.transforms.RandomPerspective(distortion_scale=0.5, p=0.5),  # 随机网格畸变
            torchvision.transforms.ToTensor(),  # 转换为张量
            torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])  # 归一化
        ])
        transform_test = torchvision.transforms.Compose([
            torchvision.transforms.Resize((256, 256)),  # 缩放到 40x40 像素
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                             [0.2023, 0.1994, 0.2010])])
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
        training_loader = torch.utils.data.DataLoader(train_ds, batch_size=4, shuffle=True, pin_memory=True,
                                                      pin_memory_device='cuda' if torch.cuda.is_available() else '',
                                                      drop_last=True)
        validation_loader = torch.utils.data.DataLoader(valid_ds, batch_size=4, shuffle=False, pin_memory=True,
                                                        pin_memory_device='cuda' if torch.cuda.is_available() else '',
                                                        drop_last=True)
        frequency = 100

    elif data == 'FashionMNIST':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize((256, 256)),
             transforms.Normalize((0.5,), (0.5,))])
        training_set = torchvision.datasets.FashionMNIST('./data', train=True, transform=transform, download=True)
        validation_set = torchvision.datasets.FashionMNIST('./data', train=False, transform=transform, download=True)

        # Create data loaders for our datasets; shuffle for training, not for validation
        training_loader = torch.utils.data.DataLoader(training_set, batch_size=4, shuffle=True, pin_memory=True,
                                                      pin_memory_device='cuda' if torch.cuda.is_available() else '')
        validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=4, shuffle=False, pin_memory=True,
                                                        pin_memory_device='cuda' if torch.cuda.is_available() else '')
        frequency = 1000

    return training_loader, validation_loader,frequency


if __name__ == '__main__':
    "----------------------------------- data ---------------------------------------------"
    training_loader, validation_loader,frequency = dataSelector('Breast')
    # training_loader,validation_loader,frequency = dataSelector('FashionMNIST')

    "----------------------------------- loss function ---------------------------------------------"
    loss_fn = torch.nn.CrossEntropyLoss()

    "----------------------------------- model ---------------------------------------------"
    num_class = 6
    model, writer, optimizer, timestamp = modelSelector('GoogleNet', 0.001, num_class)

    "----------------------------------- training ---------------------------------------------"
    epoch_number = 0

    EPOCHS = 100

    best_vloss = 1_000_000.

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss, avg_precision, avg_f1 = train_one_epoch(epoch_number, writer)

        running_vloss = 0.0
        running_vprecison = 0.0
        running_vf1 = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata
                if torch.cuda.is_available():
                    vinputs = vinputs.to(torch.device('cuda'))
                    vlabels = vlabels.to(torch.device('cuda'))
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                vprecision = multiclass_precision(voutputs, vlabels).tolist()
                vf1 = multiclass_f1_score(voutputs, vlabels, average='macro', num_classes=num_class).tolist()
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

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'model_{}_{}'.format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)

        epoch_number += 1
