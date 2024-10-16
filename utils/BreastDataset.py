import os
import pandas as pd
import numpy as np
from logging import warning
import torchvision.transforms
from utils.TableDataset import TableDataset
from typing import *
import re
from logging import debug


def make_table(data_folder_path: str, 
               official_train: bool = True, 
               BUS: bool = True, 
               USG: bool = True,
               trainROI: bool = False,
               fea_official_train=False, *, seed: int = 42) -> pd.DataFrame:
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
    trainROI_data_path = os.path.join(data_folder_path, 'breast', 'cla', 'trainROI')
    fea_official_data_path = os.path.join(data_folder_path, 'breast', 'fea', 'train')
    assert os.path.exists(official_data_path), f"{official_data_path} does not exist! please use OfficialClaDataOrganizer first!"
    assert os.path.exists(BUS_data_path), f"{BUS_data_path} does not exist! please run replace.ipynb first!"
    assert os.path.exists(USG_data_path), f"{USG_data_path} does not exist! please run process.ipynb first!"
    assert os.path.exists(fea_official_data_path), f"{fea_official_data_path} does not exist! please run OfficialFeaDataOrganizer first!"
    if fea_official_train:
        assert not (official_train or BUS or USG), "不能同时选择cla数据和fea数据"
    tables = []
    if official_train:
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
    if fea_official_train:
        table = pd.read_csv(os.path.join(fea_official_data_path, 'ground_truth.csv'),dtype=str)
        table.file_name = table.file_name.apply(lambda x: os.path.join(fea_official_data_path, x))
        tables.append(table)
    if trainROI:
        table = pd.read_csv(os.path.join(trainROI_data_path, 'ground_truth.csv'),dtype=str)
        table['label'] = table['label'].apply(lambda x: int(x))
        table.file_name = table.file_name.apply(lambda x: os.path.join(trainROI_data_path, x))
        tables.append(table)


    assert len(tables), "No selected dataset!"
    table = pd.concat(tables, axis=0)
    table.reset_index(drop=True, inplace=True)
    idx = np.arange(len(table))
    np.random.default_rng(seed).shuffle(idx)
    table = table.iloc[idx, :]
    return table.reset_index(drop=True)


def split_augmented_image(valid_dataset: pd.DataFrame,
                          augmented_folder_list: Optional[List[str]] = None) -> pd.DataFrame:
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
        augmented_ground_truth = pd.read_csv(os.path.join(augmented_folder, "ground_truth.csv"))
        # print(augmented_ground_truth)
        augmented_ground_truth.file_name = augmented_ground_truth.file_name.apply(
            lambda x: os.path.join(augmented_folder, x))
        augmented_ground_truths.append(augmented_ground_truth)
    augmented_image_table = pd.concat(augmented_ground_truths, axis=0)
    mask = ~augmented_image_table.file_name.apply(lambda x: in_valid(x, taboo_list))
    return augmented_image_table.loc[mask, :]


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


class BreastCrossValidationData:
    def __init__(self, data_folder_path: str, k_fold: int = 5,
                 train_transform: Optional[torchvision.transforms.Compose] = None,
                 valid_transform: Optional[torchvision.transforms.Compose] = None, image_format: str = 'PIL',
                 official_train: bool = True, BUS: bool = True, USG: bool = True, trainROI:bool=False,
                 fea_official_train=False, *,
                 seed: int = 42,
                 augmented_folder_list: Optional[List[str]] = None, **kwargs):
        """
        初始化返回k折叫交叉验证数据集的迭代器
        :param data_folder_path:
        :param k_fold:
        :param train_transform:
        :param valid_transform:
        :param image_format:
        :param BUS:
        :param USG:
        :param trainROI:
        :param fea_official_train:
        :param seed:
        :param augmented_folder_list: 增强后的图像所在文件夹的完整路径！
        """
        self.table = make_table(data_folder_path=data_folder_path, official_train=official_train, BUS=BUS, USG=USG,
                                trainROI=trainROI, fea_official_train=fea_official_train, seed=seed)
        self.sep_point = np.round(np.linspace(0, self.table.shape[0], k_fold + 1)).astype(np.int_)
        self.cur_valid_fold = 0
        self.k_fold = k_fold
        self.image_format = image_format
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.augmented_folder_list = augmented_folder_list
        self.seed = seed

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
                                         split_augmented_image(valid_table, self.augmented_folder_list).sample(frac=1,
                                                                                                               random_state=self.seed)])
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
                            official_train: bool = True,
                            BUS: bool = True, 
                            USG: bool = True, 
                            trainROI: bool = False,
                            fea_official_train=False, 
                            image_format: str = 'PIL', *,
                            seed: int = 42,
                            augmented_folder_list: Optional[List[str]] = None, **kwargs) -> Optional[Tuple[TableDataset, TableDataset]]:
    """
    返回单折按照给定比例划分的训练集和验证集，用yield返回是为了可以和CV一样用for train_ds,valid_ds in dataset一样来获取
    :param data_folder_path:
    :param valid_ratio:
    :param train_transform:
    :param valid_transform:
    :param BUS:
    :param USG:
    :param trainROI:
    :param fea_official_train:
    :param image_format:
    :param seed:
    :param augmented_folder_list:
    :return:
    """
    table = make_table(data_folder_path=data_folder_path, official_train=official_train, BUS=BUS, USG=USG,
                        trainROI=trainROI, fea_official_train=fea_official_train, seed=seed)
    sep_point = int(table.shape[0] * valid_ratio)
    valid_table = table.iloc[:sep_point, :]
    train_table = table.iloc[sep_point:, :]
    if augmented_folder_list:
        train_table = pd.concat(
            [train_table, split_augmented_image(valid_table, augmented_folder_list).sample(frac=1, random_state=seed)])
    debug(f"single fold sample distribution:")
    debug(f"train:")
    debug(train_table.label.value_counts())
    debug(f"valid:")
    debug(valid_table.label.value_counts())
    debug(f"num_train:{len(train_table)}, num_valid:{len(valid_table)}")
    yield (
        TableDataset(train_table, transform=train_transform, image_format=image_format),
        TableDataset(valid_table, transform=valid_transform, image_format=image_format))
