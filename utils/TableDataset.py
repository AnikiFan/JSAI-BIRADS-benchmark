import os
from torchvision.datasets import VisionDataset
import pandas as pd
import numpy as np
from torchvision.transforms import transforms
from PIL import Image
class TableDataset(VisionDataset):
    toTensor = transforms.ToTensor()

    def __init__(self, table:pd.DataFrame, transform=None, image_format='PIL'):
        """
        基于DataFrame(CSV)的数据集类
        :param table: 必须包含两列：file_name存储图像的完整路径，即可以直接读取的路径，label存储对应样本的标签
        :param transform:
        :param image_format:
        """
        super().__init__()
        self.table = table
        if image_format == 'PIL':
            self.reader = TableDataset._PIL_reader
        elif image_format == 'Tensor':
            toTensor = transforms.ToTensor()
            self.reader = TableDataset._Tensor_reader
        elif image_format == 'ndarray':
            self.reader = TableDataset._ndarray_reader
        self.classes = self.table.label.unique()
        self.labels = self.table.label
        self.transform = transform
        self.idx = 0
        self.samples = self

    @staticmethod
    def _PIL_reader(image_path):
        """
        返回PIL格式的图片读取函数
        :param image_path:
        :return:
        """
        return Image.open(image_path).convert("RGB")

    @staticmethod
    def _Tensor_reader(image_path):
        """
        返回Tensor格式的图片读取函数
        :param image_path:
        :return:
        """
        return TableDataset.toTensor(np.array(Image.open(image_path).convert('RGB')))

    @staticmethod
    def _ndarray_reader(image_path):
        """
        返回ndarray格式的图片读取函数
        :param image_path:
        :return:
        """
        return np.array(Image.open(image_path).convert('RGB'))

    def __len__(self):
        """
        返回样本个数
        :return:
        """
        return self.table.shape[0]

    def __getitem__(self, item):
        """
        指定下标，从0开始，返回对应的图像和label
        :param item:
        :return:
        """
        if self.transform:
            return self.transform(self.reader(self.table.iloc[item].file_name)), self.table.iloc[item].label
        return self.reader(self.table.iloc[item].file_name), self.table.iloc[item].label

    def __iter__(self):
        """
        使该类支持迭代器方式的使用
        :return:
        """
        return self

    def __next__(self):
        """
        使该类支持迭代器方式的使用
        :return:
        """
        if self.idx < len(self):
            self.idx += 1
            return self[self.idx - 1]
        else:
            raise StopIteration
    # def __getitems__(self, items):
    #     return self.table.iloc[items, 'file_name'].apply(lambda x: self.reader(x)).to_list()