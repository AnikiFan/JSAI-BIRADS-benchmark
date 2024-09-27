import os
from torchvision.datasets import VisionDataset
import pandas as pd
import numpy as np
from torchvision.transforms import transforms
from PIL import Image
class TableDataset(VisionDataset):
    toTensor = transforms.ToTensor()

    def __init__(self, table, transform=None, image_format='PIL'):
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
        return Image.open(image_path).convert("RGB")

    @staticmethod
    def _Tensor_reader(image_path):
        return TableDataset.toTensor(np.array(Image.open(image_path).convert('RGB')))

    @staticmethod
    def _ndarray_reader(image_path):
        return np.array(Image.open(image_path).convert('RGB'))

    def __len__(self):
        return self.table.shape[0]

    def __getitem__(self, item):
        if self.transform:
            return self.transform(self.reader(self.table.iloc[item].file_name)), self.table.iloc[item].label
        return self.reader(self.table.iloc[item].file_name), self.table.iloc[item].label

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx < len(self):
            self.idx += 1
            return self[self.idx - 1]
        else:
            raise StopIteration
    # def __getitems__(self, items):
    #     return self.table.iloc[items, 'file_name'].apply(lambda x: self.reader(x)).to_list()