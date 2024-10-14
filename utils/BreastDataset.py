import os
from http.client import UnimplementedFileMode

import pandas as pd
import numpy as np
from logging import warning
import torchvision.transforms
from utils.TableDataset import TableDataset
from typing import *
import re
from logging import debug


def make_table(data_folder_path: str, official_train: bool = True, BUS: bool = True, USG: bool = True,
               fea_official_train=False, feature='all', *, seed: int = 42) -> pd.DataFrame:
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
    fea_official_data_path = os.path.join(data_folder_path, 'breast', 'fea', 'train')
    assert os.path.exists(
        official_data_path), f"{official_data_path} does not exist! please use OfficialClaDataOrganizer first!"
    assert os.path.exists(BUS_data_path), f"{BUS_data_path} does not exist! please run replace.ipynb first!"
    assert os.path.exists(USG_data_path), f"{USG_data_path} does not exist! please run process.ipynb first!"
    assert os.path.exists(
        fea_official_data_path), f"{fea_official_data_path} does not exist! please run OfficialFeaDataOrganizer first!"
    if fea_official_train:
        assert not (official_train or BUS or USG), "不能同时选择cla数据和fea数据"
    tables = []
    if official_train:
        table = pd.read_csv(os.path.join(official_data_path, 'ground_truth.csv'))
        table.file_name = table.file_name.apply(lambda x: os.path.join(official_data_path, x))
        tables.append(table)
        debug("append official_train")
    if BUS:
        table = pd.read_csv(os.path.join(BUS_data_path, 'ground_truth.csv'))
        table.file_name = table.file_name.apply(lambda x: os.path.join(BUS_data_path, x))
        tables.append(table)
        debug("append BUS")
    if USG:
        table = pd.read_csv(os.path.join(USG_data_path, 'ground_truth.csv'))
        table.file_name = table.file_name.apply(lambda x: os.path.join(USG_data_path, x))
        tables.append(table)
        debug("append USG")
    if fea_official_train:
        table = pd.read_csv(os.path.join(fea_official_data_path, 'ground_truth.csv'), dtype=str)
        table.file_name = table.file_name.apply(lambda x: os.path.join(fea_official_data_path, x))
        tables.append(table)
        debug("append fea_official_train")
    assert len(tables), "No selected dataset!"
    table = pd.concat(tables, axis=0)
    table.reset_index(drop=True, inplace=True)
    idx = np.arange(len(table))
    np.random.default_rng(seed).shuffle(idx)
    table = table.iloc[idx, :]
    if fea_official_train:
        if feature == 'boundary':
            table.label = table.label.str[0].astype(np.int64)
            debug("only use boundary feature")
        elif feature == 'calcification':
            table.label = table.label.str[1].astype(np.int64)
            debug("only use calcification feature")
        elif feature == 'direction':
            table.label = table.label.str[2].astype(np.int64)
            debug("only use direction feature")
        elif feature == 'shape':
            table.label = table.label.str[3].astype(np.int64)
            debug("only use shape feature")
        elif feature == 'all':
            debug("use all features")
        else:
            warning("invalid feature selected!")

    return table.reset_index(drop=True)


def split_augmented_image(valid_dataset: pd.DataFrame, task,
                          augmented_folder_list: Optional[List[str]] = None, feature='all') -> pd.DataFrame:
    """
    根据在未经变换的数据集上得到的验证集，将经过数据增广的数据划分为
    1. 不能加入训练集
    2. 可以加入训练集
    并将第2类合并为dataframe返回
    在判断增广前的图像是否在验证集中时，区分文件后缀，即即使文件名相同，一个时jpg，一个是png，允许一个在验证集，一个在训练集中
    :param valid_dataset: 已划为验证集的图像对应csv，file_name应该包含完整路径
    :param augmented_folder_list: 增强后的图像所在文件夹
    :return: 可以加入训练集的图像组成的dataframe
    """
    taboo_list = valid_dataset.file_name.str.split(os.sep).str[-1].tolist()
    assert 'file_name' in valid_dataset.columns, 'valid dataset should have column "file_name"'
    assert 'label' in valid_dataset.columns, 'valid dataset should have column "label"'
    augmented_ground_truths = []
    if not augmented_folder_list:
        warning("empty augmented folder list !")
        return pd.DataFrame()
    for augmented_folder in augmented_folder_list:
        if not os.path.exists(augmented_folder):
            warning(f"augmented image folder {augmented_folder} doesn't exist!")
            continue
        if task == 'fea':
            augmented_ground_truth = pd.read_csv(os.path.join(augmented_folder, "ground_truth.csv"), dtype=str)
        else:
            augmented_ground_truth = pd.read_csv(os.path.join(augmented_folder, "ground_truth.csv"))
        debug(f"using augmented data in folder {augmented_folder}")
        # print(augmented_ground_truth)
        augmented_ground_truth.file_name = augmented_ground_truth.file_name.apply(
            lambda x: os.path.join(augmented_folder, x))
        augmented_ground_truths.append(augmented_ground_truth)
    augmented_image_table = pd.concat(augmented_ground_truths, axis=0)
    mask = ~augmented_image_table.file_name.apply(lambda x: in_valid(x, taboo_list))
    augmented_image_table = augmented_image_table.loc[mask, :]
    if feature == 'boundary':
        augmented_image_table.label = augmented_image_table.label.str[0].astype(np.int64)
        debug("only use boundary feature in augmented image")
    elif feature == 'calcification':
        augmented_image_table.label = augmented_image_table.label.str[1].astype(np.int64)
        debug("only use calcification feature in augmented image")
    elif feature == 'direction':
        augmented_image_table.label = augmented_image_table.label.str[2].astype(np.int64)
        debug("only use direction feature in augmented image")
    elif feature == 'shape':
        augmented_image_table.label = augmented_image_table.label.str[3].astype(np.int64)
        debug("only use shape feature in augmented image")
    elif feature == 'all':
        debug("use all features in augmented image")
    else:
        warning("invalid feature selected!")
    return augmented_image_table


def in_valid(file_name: str, valid_list: List[str]) -> bool:
    """
    检验file_name对应的文件能否加入训练集
    :param file_name: 待检验图像的完整路径
    :param valid_list: 由验证集中的图像的文件名，包含后缀名，不包含完整路径，组成的列表
    :return:
    """
    file_name = file_name.split(os.sep)[-1]
    file_name = re.sub(r'\s*\(\d+\)\s*(?=\.\w+)', '', file_name)  # 获取文件名，去除(d)
    if '__mixup__' not in file_name:
        return file_name in valid_list
    file_name1, file_name2 = file_name[:-4].split('__mixup__')
    return file_name1 in valid_list or file_name2 in valid_list


def adjust_ratio(train: pd.DataFrame, ratio: str | Tuple[float] | Tuple[int]) -> pd.DataFrame:
    debug('original train distribution:')
    debug(train.label.value_counts())
    if ratio == 'same':
        debug('keep original distribution')
        return train
    else:
        assert len(
            ratio) == train.label.nunique(), f"there are {train.label.nunique()} different labels, but the length of ratio is {len(ratio)}"
        ratio = np.array(ratio)
        relative = train.label.value_counts(sort=False) / ratio
        keep_time = relative.min()
        final_distribution = np.floor(keep_time * ratio).astype(np.int64)
        debug("expected final distribution:")
        debug(final_distribution)
        tables = []
        for idx, label in enumerate(np.sort(np.unique(train.label))):
            tables.append(train[train['label'] == label].iloc[:final_distribution[idx], :])
        table = pd.concat(tables, axis=0).sort_index()
        debug("actual final distribution")
        debug(table.label.value_counts())
        return table


class BreastCrossValidationData:
    def __init__(self, data_folder_path: str, k_fold: int = 5,
                 train_transform: Optional[torchvision.transforms.Compose] = None,
                 valid_transform: Optional[torchvision.transforms.Compose] = None, image_format: str = 'PIL',
                 official_train: bool = True, BUS: bool = True, USG: bool = True, fea_official_train=False,
                 feature='all', ratio: str | Tuple[float] | Tuple[int] = 'same', *,
                 seed: int = 42,
                 augmented_folder_list: Optional[List[str]] = None, **kwargs):
        """
        初始化返回k折叫交叉验证数据集的迭代器
        :param data_folder_path:
        :param k_fold:
        :param train_transform:
        :param valid_transform:
        :param image_format:
        :param official_train:
        :param BUS:
        :param USG:
        :param fea_official_train:
        :param feature: 选择使用fea任务的某个特征进行训练，默认为'all'，即使用全部
        :param ratio: 指定训练集中的标签比例，若为'same'，即保持不变
        :param seed:
        :param augmented_folder_list: 增强后的图像所在文件夹的完整路径！
        :param kwargs:
        """
        self.table = make_table(data_folder_path=data_folder_path, official_train=official_train, BUS=BUS, USG=USG,
                                fea_official_train=fea_official_train, feature=feature, seed=seed)
        self.sep_point = np.round(np.linspace(0, self.table.shape[0], k_fold + 1)).astype(np.int_)
        self.cur_valid_fold = 0
        self.k_fold = k_fold
        self.image_format = image_format
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.augmented_folder_list = augmented_folder_list
        self.seed = seed
        self.ratio = ratio
        self.feature = feature
        self.task = 'fea' if fea_official_train else 'cla'

    def __len__(self) -> int:
        return self.k_fold

    def __iter__(self):
        return self

    def __next__(self) -> Optional[Tuple[TableDataset, TableDataset]]:
        if self.cur_valid_fold < self.k_fold:
            self.cur_valid_fold += 1
            valid_table = self.table.iloc[self.sep_point[self.cur_valid_fold - 1]:
                                          self.sep_point[self.cur_valid_fold], :]
            train_table = pd.concat([self.table.iloc[:self.sep_point[self.cur_valid_fold - 1], :],
                                     self.table.iloc[self.sep_point[self.cur_valid_fold]:, :]])
            if self.augmented_folder_list:
                train_table = pd.concat([train_table,
                                         split_augmented_image(valid_table, task=self.task,augmented_folder_list= self.augmented_folder_list,
                                                feature=self.feature).sample(frac=1,random_state=self.seed)])
            train_table = adjust_ratio(train_table, self.ratio)
            debug(f"fold {self.cur_valid_fold - 1} sample distribution:")
            debug(f"train:")
            debug(train_table.label.value_counts())
            debug(f"valid:")
            debug(valid_table.label.value_counts())
            debug(f"num_train:{len(train_table)}, num_valid:{len(valid_table)}")
            train_dataset = TableDataset(train_table, transform=self.train_transform, image_format=self.image_format)
            valid_dataset = TableDataset(valid_table, transform=self.valid_transform, image_format=self.image_format)
            return train_dataset, valid_dataset
        raise StopIteration


def getBreastTrainValidData(data_folder_path: str, valid_ratio: float = 0.2,
                            train_transform: Optional[torchvision.transforms.Compose] = None,
                            valid_transform: Optional[torchvision.transforms.Compose] = None,
                            official_train: bool = True, BUS: bool = True, USG: bool = True, fea_official_train=False,
                            feature='all', image_format: str = 'PIL', ratio: str | Tuple[float] | Tuple[int] = 'same',
                            *, seed: int = 42,
                            augmented_folder_list: Optional[List[str]] = None, **kwargs) -> Optional[
    Tuple[TableDataset, TableDataset]]:
    """
    返回单折按照给定比例划分的训练集和验证集，用yield返回是为了可以和CV一样用for train_ds,valid_ds in dataset一样来获取
    :param data_folder_path:
    :param valid_ratio:
    :param train_transform:
    :param valid_transform:
    :param official_train:
    :param BUS:
    :param USG:
    :param fea_official_train:
    :param feature: 选择使用fea任务的某个特征进行训练，默认为'all'，即使用全部
    :param image_format:
    :param ratio: 指定训练集中的标签比例，若为'same'，即保持不变
    :param seed:
    :param augmented_folder_list:
    :param kwargs:
    :return:
    """
    table = make_table(data_folder_path=data_folder_path, official_train=official_train, BUS=BUS, USG=USG,
                       fea_official_train=fea_official_train, feature=feature, seed=seed)
    sep_point = int(table.shape[0] * valid_ratio)
    valid_table = table.iloc[:sep_point, :]
    train_table = table.iloc[sep_point:, :]
    if augmented_folder_list:
        train_table = pd.concat(
            [train_table, split_augmented_image(valid_table, task='fea' if fea_official_train else 'cla',
                augmented_folder_list=augmented_folder_list, feature=feature).sample(frac=1, random_state=seed)])
    train_table = adjust_ratio(train_table, ratio)
    debug(f"single fold sample distribution:")
    debug(f"train:")
    debug(train_table.label.value_counts())
    debug(f"valid:")
    debug(valid_table.label.value_counts())
    debug(f"num_train:{len(train_table)}, num_valid:{len(valid_table)}")
    yield (
        TableDataset(train_table, transform=train_transform, image_format=image_format),
        TableDataset(valid_table, transform=valid_transform, image_format=image_format))
