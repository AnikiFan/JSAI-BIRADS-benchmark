import torch
from torchvision.datasets import VisionDataset
import pandas as pd
import numpy as np
from torchvision.transforms import transforms
from PIL import Image
from typing import *
import os


class ROIDataset(VisionDataset):
    def __init__(self, data_foler_path, size):
        super(ROIDataset, self).__init__()
        self.table = pd.read_csv(os.path.join(data_foler_path, 'ground_truth.csv'))
        self.data_folder_path = data_foler_path
        self.idx = 0
        self.samples = self
        self.to_tensor = transforms.ToTensor()
        self.size = size

    def __len__(self) -> int:
        """
        返回样本个数
        :return:
        """
        return self.table.shape[0]

    def __getitem__(self, item: int) -> Tuple[Any, Any]:
        """
        指定下标，从0开始，返回对应的图像和label
        :param item:
        :return:
        """
        label = self.table.iloc[item].label
        file_name = self.table.iloc[item].file_name
        file_path = os.path.join(self.data_folder_path, file_name)
        masked_img = Image.open(file_path.replace('train', 'masked_train').replace('test', 'masked_test')).convert('L').resize(self.size)
        img = Image.open(file_path).convert('L').resize(self.size)
        masked_img = self.to_tensor(masked_img)
        img = self.to_tensor(img)
        output = torch.cat([img,masked_img, img],dim=0)
        return output, label

    def __iter__(self):
        """
        使该类支持迭代器方式的使用
        :return:
        """
        return self

    def __next__(self) -> Optional[Tuple[Any, Any]]:
        """
        使该类支持迭代器方式的使用
        :return:
        """
        if self.idx < len(self):
            self.idx += 1
            return self[self.idx - 1]
        else:
            raise StopIteration


def getROIBreastTrainValidData(data_folder_path, **kwargs):
    yield ROIDataset(os.path.join(data_folder_path, 'breast', 'cla', 'train'), size=(224, 224)), ROIDataset(
        os.path.join(data_folder_path, 'breast', 'cla', 'test'), size=(224, 224))


if __name__ == '__main__':
    dataset = ROIDataset(os.path.join(os.pardir, 'data', 'breast', 'cla', 'test'), (224, 224))
    print(len(dataset))
    sample, label = dataset[0]
    print(sample.shape)
    print(type(sample))
    print(label)
    print(np.unique(sample.numpy()))
    train,valid = next(getROIBreastTrainValidData(os.path.join(os.pardir,'data')))
    sample, label = train[0]
    print(sample.shape)
    print(type(sample))
    print(label)
    sample, label = valid[0]
    print(sample.shape)
    print(type(sample))
    print(label)



