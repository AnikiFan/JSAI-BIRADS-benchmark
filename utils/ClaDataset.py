import os
from torchvision.datasets import VisionDataset
import pandas as pd
import numpy as np
from torchvision.transforms import transforms
from PIL import Image


def make_table(data_folder_path, official=True, BUS=True, USG=True):
    official_data_path = os.path.join(data_folder_path, 'breast', 'myTrain', 'cla')
    BUS_data_path = os.path.join(data_folder_path, 'breast', 'BUS', 'Images')
    USG_data_path = os.path.join(data_folder_path, 'breast', 'USG')
    assert os.path.exists(official_data_path), "please run claTrainSetOrganize.py first!"
    assert os.path.exists(BUS_data_path), "please run replace.ipynb first!"
    assert os.path.exists(USG_data_path), "please run process.ipynb first!"
    tables = []
    table = pd.read_csv(os.path.join(official_data_path, 'ground_truth.csv'), index_col=0)
    table.file_name = table.file_name.apply(lambda x: os.path.join(official_data_path, x))
    tables.append(table)
    if BUS:
        table = pd.read_csv(os.path.join(BUS_data_path, 'ground_truth.csv'), index_col=0)
        table.file_name = table.file_name.apply(lambda x: os.path.join(BUS_data_path, x))
        tables.append(table)
    if USG:
        table = pd.read_csv(os.path.join(USG_data_path, 'ground_truth.csv'), index_col=0)
        table.file_name = table.file_name.apply(lambda x: os.path.join(USG_data_path, x))
        tables.append(table)
    table = pd.concat(tables, axis=0)
    table.reset_index(drop=True, inplace=True)
    return table


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

    # def __getitems__(self, items):
    #     return self.table.iloc[items, 'file_name'].apply(lambda x: self.reader(x)).to_list()


class ClaCrossValidationData:
    def __init__(self, data_folder_path, k_fold=5, train_transform=None, valid_transform=None, image_format='PIL',
                 BUS=True, USG=True, seed=42):
        self.table = make_table(data_folder_path=data_folder_path, official=True, BUS=BUS, USG=USG)
        idx = np.arange(self.table.shape[0])
        np.random.default_rng(seed=seed).shuffle(idx)
        self.table = self.table.iloc[idx, :]
        self.sep_point = np.round(np.linspace(0, self.table.shape[0], k_fold + 1)).astype(np.int_)
        self.cur_valid_fold = 0
        self.k_fold = k_fold
        self.image_format = image_format
        self.train_transform = train_transform
        self.valid_transform = valid_transform

    def __len__(self):
        return self.k_fold

    def __next__(self):
        if self.cur_valid_fold < self.k_fold:
            self.cur_valid_fold += 1
            return (
                TableDataset(
                    pd.concat([self.table.iloc[:self.sep_point[self.cur_valid_fold - 1], :],
                               self.table.iloc[self.sep_point[self.cur_valid_fold]:, :]]),
                    transform=self.train_transform, image_format=self.image_format),
                TableDataset(
                    self.table.iloc[self.sep_point[self.cur_valid_fold - 1]:self.sep_point[self.cur_valid_fold], :],
                    transform=self.valid_transform,
                    image_format=self.image_format)
            )
        raise StopIteration


def getClaTrainValidData(data_folder_path, valid_ratio=0.2, train_transform=None, valid_transform=None, BUS=True,
                         USG=True,
                         image_format='PIL', seed=42):
    table = make_table(data_folder_path=data_folder_path, official=True, BUS=BUS, USG=USG)
    idx = np.arange(table.shape[0])
    np.random.default_rng(seed=seed).shuffle(idx)
    table = table.iloc[idx, :]
    sep_point = int(table.shape[0] * valid_ratio)
    return (
        TableDataset(table.iloc[sep_point:, :], transform=train_transform, image_format=image_format),
        TableDataset(table.iloc[:sep_point, :], transform=valid_transform, image_format=image_format))


if __name__ == "__main__":
    test = ClaCrossValidationData(data_folder_path=os.path.join(os.pardir, 'data'), image_format='Tensor')
    for i in range(5):
        train, valid = next(test)
        print(len(train), len(valid))
        print(type(train[0][0]), type(valid[0][0]))
        print(type(train[0][1]), type(valid[0][1]))
        print(train[0][0].shape, valid[0][0].shape)
        print(train[0][1], valid[0][1])
