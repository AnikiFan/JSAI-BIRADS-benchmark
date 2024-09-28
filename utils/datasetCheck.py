import torch
from collections import Counter
import matplotlib.pyplot as plt
import logging
from ClaDataset import getClaTrainValidData
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from ClaDataset import ClaCrossValidationData

logging.getLogger('matplotlib.font_manager').disabled = True


def checkDataset(train_ds, valid_ds, training_loader, validation_loader, num_samples_to_show=4):
    """
    检查数据集格式的函数，包括类别数量、类别分布、数据类型、数据维度等。

    参数:
    - train_ds: 训练集 (torchvision.datasets.ImageFolder)
    - valid_ds: 验证集 (torchvision.datasets.ImageFolder)
    - training_loader: 训练数据加载器 (torch.utils.data.DataLoader)
    - validation_loader: 验证数据加载器 (torch.utils.data.DataLoader)
    - num_samples_to_show: 可视化的样本数量，默认是4

    返回:
    - 无返回值，但会打印检查结果并显示样本图像
    """

    # 检查类别数量
    num_classes = len(train_ds.classes)
    print(f"Detected number of classes: {num_classes}")

    # 检查每个类别的样本数量
    train_labels = [label for _, label in tqdm(train_ds.samples)]
    train_counter = Counter(train_labels)
    print("Train dataset class distribution:", train_counter)

    valid_labels = [label for _, label in tqdm(valid_ds.samples)]
    valid_counter = Counter(valid_labels)
    print("Validation dataset class distribution:", valid_counter)

    # 检查训练集中第一张图像的类型和尺寸
    sample_image, sample_label = train_ds[0]

    # 确保训练集图像张量是 FloatTensor 并且尺寸正确
    assert isinstance(sample_image, torch.FloatTensor), "Train image is not a FloatTensor"
    assert sample_image.size(0) == 3, "Train image: Expected 3 channels (RGB)"

    print(f"Train sample image shape: {sample_image.size()}")
    print(f"Train sample image type: {type(sample_image)}")
    print(f"Train sample image label: {sample_label}")

    # 确保验证集图像张量是 FloatTensor 并且尺寸正确
    valid_image, valid_label = valid_ds[0]
    assert isinstance(valid_image, torch.FloatTensor), "Validation image is not a FloatTensor"
    assert valid_image.size(0) == 3, "Validation image: Expected 3 channels (RGB)"

    # 检查验证集中第一张图像的类型和尺寸
    print(f"Validation sample image shape: {valid_image.size()}")
    print(f"Validation sample image type: {type(valid_image)}")
    print(f"Validation sample image label: {valid_label}")

    # 检查训练数据加载器的批次
    for images, labels in training_loader:
        print(f"Training batch images shape: {images.size()}")  # 通常 [batch_size, 3, H, W]
        print(f"Training batch labels shape: {labels.size()}")  # 通常 [batch_size]
        print(f"Training batch images dtype: {images.dtype}")
        print(f"Training batch labels dtype: {labels.dtype}")
        break  # 只检查第一个批次

    # 检查验证数据加载器的批次
    for images, labels in validation_loader:
        print(f"Validation batch images shape: {images.size()}")  # 通常 [batch_size, 3, H, W]
        print(f"Validation batch labels shape: {labels.size()}")  # 通常 [batch_size]
        print(f"Validation batch images dtype: {images.dtype}")
        print(f"Validation batch labels dtype: {labels.dtype}")
        break  # 只检查第一个批次

    # 显示一些转换后的图像
    def show_images(images, labels, title):
        plt.figure(figsize=(10, 10))
        for i in range(len(images)):
            plt.subplot(1, len(images), i + 1)
            plt.imshow(images[i].permute(1, 2, 0))  # 调整为 HWC 形式以便于展示
            plt.title(f"Label: {labels[i]}")
            plt.axis('off')
        plt.suptitle(title)
        plt.show()

    if num_samples_to_show > 0:
        # 从训练加载器中取样本展示
        train_samples, train_labels = next(iter(training_loader))
        show_images(train_samples[:num_samples_to_show], train_labels[:num_samples_to_show], "Training Samples")

        # 从验证加载器中取样本展示
        valid_samples, valid_labels = next(iter(validation_loader))
        show_images(valid_samples[:num_samples_to_show], valid_labels[:num_samples_to_show], "Validation Samples")

    return


# 使用示例
# 调用该函数来检查你的数据集格式
# check_dataset_format(train_ds, valid_ds, training_loader, validation_loader)
if __name__ == '__main__':
    augmented_folder_path = os.path.join(os.pardir, 'data', 'breast', 'cla', 'augmented')
    CVData = ClaCrossValidationData(data_folder_path=os.path.join(os.pardir, 'data'), image_format='Tensor',
            augmented_folder_list=[os.path.join(augmented_folder_path, x) for x in os.listdir(augmented_folder_path)])
    # train_ds, valid_ds = getClaTrainValidData(data_folder_path=os.path.join(os.pardir, 'data'), image_format='Tensor',
    #                                           augmented_folder_list=[os.path.join(augmented_folder_path, x) for x in
    #                                                                  os.listdir(augmented_folder_path)])
    # train_loader, valid_loader = DataLoader(train_ds), DataLoader(valid_ds)
    for train_ds,valid_ds in iter(CVData):
        train_loader,valid_loader = DataLoader(train_ds),DataLoader(valid_ds)
        checkDataset(train_ds, valid_ds, train_loader, valid_loader,num_samples_to_show=0)
        print()
        print()
