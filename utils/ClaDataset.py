import os
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from torchvision.io import decode_image
from torchvision.transforms import transforms
from PIL import Image

official_data_path = os.path.join(os.pardir, 'data', 'breast', 'myTrain', 'cla')
BUS_data_path = os.path.join(os.pardir, 'data', 'breast', 'BUS', 'Images')
USG_data_path = os.path.join(os.pardir, 'data', 'breast', 'USG')


def make_table(official=True, BUS=True, USG=True):
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


class TableDataset(Dataset):
    def __init__(self, table, image_format='PIL'):
        super().__init__()
        self.table = table
        if image_format == 'PIL':
            self.reader = lambda x: Image.open(x).convert("RGB")
        elif image_format == 'Tensor':
            toTensor = transforms.ToTensor()
            self.reader = lambda x:toTensor(np.array(Image.open(x).convert('RGB')))
        elif image_format == 'ndarray':
            self.reader = lambda x: np.array(Image.open(x).convert('RGB'))

    def __len__(self):
        return self.table.shape[0]

    def __getitem__(self, item):
        return self.reader(self.table.iloc[item].file_name), self.table.iloc[item].label

    # def __getitems__(self, items):
    #     return self.table.iloc[items, 'file_name'].apply(lambda x: self.reader(x)).to_list()


class ClaCrossValidationData:
    def __init__(self, k_fold=5, BUS=True, USG=True, image_format='PIL', seed=42):
        self.table = make_table(official=True, BUS=BUS, USG=USG)
        idx = np.arange(self.table.shape[0])
        np.random.default_rng(seed=seed).shuffle(idx)
        self.table = self.table.iloc[idx, :]
        self.sep_point = np.round(np.linspace(0, self.table.shape[0], k_fold + 1)).astype(np.int_)
        self.cur_valid_fold = 0
        self.k_fold = k_fold
        self.image_format = image_format

    def __len__(self):
        return self.k_fold

    def __next__(self):
        if self.cur_valid_fold < self.k_fold:
            self.cur_valid_fold += 1
            return (
                TableDataset(
                    pd.concat([self.table.iloc[:self.sep_point[self.cur_valid_fold - 1], :],
                               self.table.iloc[self.sep_point[self.cur_valid_fold]:, :]]), self.image_format),
                TableDataset(
                    self.table.iloc[self.sep_point[self.cur_valid_fold - 1]:self.sep_point[self.cur_valid_fold], :],
                    self.image_format)
            )
        raise StopIteration


if __name__ == "__main__":
    test = ClaCrossValidationData(image_format='Tensor')
    for i in range(5):
        train, valid = next(test)
        print(len(train), len(valid))
        print(type(train[0][0]),type(valid[0][0]))
        print(type(train[0][1]),type(valid[0][1]))
        print(train[0][0].shape,valid[0][0].shape)
        print(train[0][1],valid[0][1])
