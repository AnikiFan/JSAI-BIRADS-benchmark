import os
import warnings

import pandas as pd
import numpy as np
from TableDataset import TableDataset


def make_table(data_folder_path, official_train=True, BUS=True, USG=True, *, seed=42):
    """
    合成未经变换的数据整合而成的csv，file_name列为图像的路径，label列为对应图像的标签
    :param data_folder_path:
    :param official_train:
    :param BUS:
    :param USG:
    :param seed:
    :return:
    """
    official_data_path = os.path.join(data_folder_path, 'breast', 'cla', 'train')
    BUS_data_path = os.path.join(data_folder_path, 'breast', 'cla', 'BUS', 'Images')
    USG_data_path = os.path.join(data_folder_path, 'breast', 'cla', 'USG')
    assert os.path.exists(official_data_path), "please use OfficialClaDataOrganizer first!"
    assert os.path.exists(BUS_data_path), "please run replace.ipynb first!"
    assert os.path.exists(USG_data_path), "please run process.ipynb first!"
    tables = []
    table = pd.read_csv(os.path.join(official_data_path, 'ground_truth.csv'))
    table.file_name = table.file_name.apply(lambda x: os.path.join(official_data_path, x))
    tables.append(table)
    if BUS:
        table = pd.read_csv(os.path.join(BUS_data_path, 'ground_truth.csv'))
        table.file_name = table.file_name.apply(lambda x: os.path.join(BUS_data_path, x))
        tables.append(table)
    if USG:
        table = pd.read_csv(os.path.join(USG_data_path, 'ground_truth.csv'))
        table.file_name = table.file_name.apply(lambda x: os.path.join(USG_data_path, x))
        tables.append(table)
    table = pd.concat(tables, axis=0)
    table.reset_index(drop=True, inplace=True)
    idx = np.arange(len(table))
    np.random.default_rng(seed).shuffle(idx)
    table = table.iloc[idx, :]
    return table


def split_augmented_image(valid_set, augmented_folder_list):
    """
    根据在未经变换的数据集上得到的验证集，将经过数据增广的数据划分为
    1. 不能加入训练集
    2. 可以加入训练集
    并将第2类合并为dataframe返回
    :param valid_set: 已划为验证集的图像对应csv，file_name应该包含完整路径
    :param augmented_folder_list: 增强后的图像所在文件夹
    :return: 可以加入训练集的图像组成的dataframe
    """
    taboo_list = valid_set.file_name.str.split(os.sep).str[-1]
    augmented_ground_truths = []
    for augmented_folder in augmented_folder_list:
        if not os.path.exists(augmented_folder):
            warnings.warn(f"augmented image folder {augmented_folder} doesn't exist!")
            continue
        augmented_ground_truth = pd.read_csv(os.path.join(augmented_folder, "ground_truth.csv"))
        augmented_ground_truth.file_name = augmented_ground_truth.file_name.apply(
            lambda x: os.path.join(augmented_folder, x))
        augmented_ground_truths.append(augmented_ground_truth)
    augmented_image_table = pd.concat(augmented_ground_truths, axis=0)
    mask = ~augmented_image_table.file_name.str.split(os.sep).str[-1].replace(r'\s*\(\d+\)\s*(?=\.\w+)', '').isin(taboo_list)
    mixup_mask = augmented_image_table.file_name.apply(
        lambda file_name: True if '__mixup__' not in file_name else
        not file_name[:-4].split('__mixup__')[0] in taboo_list and
        not file_name[:-4].split('__mixup__')[1].replace(r'\s*\(\d+\)\s*(?=\.\w+)', '') in taboo_list)
    return augmented_image_table.loc[mask&mixup_mask, :]


class ClaCrossValidationData:
    def __init__(self, data_folder_path, k_fold=5, train_transform=None, valid_transform=None, image_format='PIL',
                 BUS=True, USG=True, *, seed=42, augmented_folder_list=None):
        """
        初始化返回k折叫交叉验证数据集的迭代器
        :param data_folder_path:
        :param k_fold:
        :param train_transform:
        :param valid_transform:
        :param image_format:
        :param BUS:
        :param USG:
        :param seed:
        :param augmented_folder_list: 增强后的图像所在文件夹的完整路径！
        """
        self.table = make_table(data_folder_path=data_folder_path, official_train=True, BUS=BUS, USG=USG, seed=seed)
        self.sep_point = np.round(np.linspace(0, self.table.shape[0], k_fold + 1)).astype(np.int_)
        self.cur_valid_fold = 0
        self.k_fold = k_fold
        self.image_format = image_format
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.augmented_folder_list = augmented_folder_list

    def __len__(self):
        return self.k_fold

    def __iter__(self):
        return self

    def __next__(self):
        if self.cur_valid_fold < self.k_fold:
            self.cur_valid_fold += 1
            valid_table = self.table.iloc[self.sep_point[self.cur_valid_fold - 1]:
                                          self.sep_point[self.cur_valid_fold],:]
            train_table = pd.concat([self.table.iloc[:self.sep_point[self.cur_valid_fold - 1], :],
                                     self.table.iloc[self.sep_point[self.cur_valid_fold]:, :]])
            if self.augmented_folder_list:
                train_table = pd.concat([train_table, split_augmented_image(valid_table, self.augmented_folder_list)])
            train_dataset = TableDataset(train_table, transform=self.train_transform, image_format=self.image_format)
            valid_dataset = TableDataset(valid_table, transform=self.valid_transform, image_format=self.image_format)
            return train_dataset, valid_dataset
        raise StopIteration


def getClaTrainValidData(data_folder_path, valid_ratio=0.2, train_transform=None, valid_transform=None, BUS=True,
                         USG=True, image_format='PIL', *, seed=42, augmented_folder_list=None):
    table = make_table(data_folder_path=data_folder_path, official_train=True, BUS=BUS, USG=USG, seed=seed)
    sep_point = int(table.shape[0] * valid_ratio)
    valid_table = table.iloc[:sep_point, :]
    train_table = table.iloc[sep_point:, :]
    if augmented_folder_list:
        train_table = pd.concat([train_table, split_augmented_image(valid_table, augmented_folder_list)])
    return (
        TableDataset(train_table, transform=train_transform, image_format=image_format),
        TableDataset(valid_table, transform=valid_transform, image_format=image_format))
