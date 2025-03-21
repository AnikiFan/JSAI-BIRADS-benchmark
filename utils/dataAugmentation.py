import os
import cv2
import numpy as np
import pandas as pd
import albumentations as A
from utils.BreastDataset import make_table
import re
from warnings import warn
from typing import *


def remove_nested_parentheses(s: str)->str:
    # 使用正则表达式匹配嵌套的圆括号及其内容
    while '(' in s:
        s = re.sub(r'\([^()]*\)', '', s)  # 移除最内层的圆括号及其内容
    return s


def make_fingerprint(transform: A.Compose, ratio: Optional[Tuple[float]])->Tuple[str,str]:
    """
    根据变换创建标识名称，一短一长，短的用于文件夹，长的写入txt中
    :param transform: 一个Aalbumentations的Compose对象！这里利用了该对象的str表示方法首尾两行为括号的特性！
    :param ratio:
    :return:
    """
    fingerprint = ''.join(str(transform).splitlines()[1:-1])
    full = fingerprint
    fingerprint = remove_nested_parentheses(fingerprint)

    if type(ratio)!=type(None):
        fingerprint += 'ratio=' + str(tuple(ratio))
    fingerprint = re.sub(r'[<>:"/\\|?*]', '_', fingerprint).replace(' ', '').rstrip(',')

    if type(ratio)!=type(None):
        full += 'ratio=' + str(tuple(ratio))
    full = re.sub(r'[<>:"/\\|?*]', '_', full).replace(' ', '').rstrip(',')
    return fingerprint, full


def make_ratio_table(table: pd.DataFrame, ratio: float) -> pd.DataFrame:
    """
    按比例形成所需的dataframe
    :param table:
    :param ratio:
    :return:
    """
    num_class = table.label.nunique()
    whole = np.round(ratio).astype(np.int_)
    left = ratio - whole
    result = []
    for label, group in table.groupby('label'):
        df = pd.concat([group.assign(no=int(i)) for i in range(1, whole[label] + 1)], axis=0)
        df = pd.concat([df, group.iloc[:round(len(group) * left[label]), :].assign(no=int(whole[label]) + 1)], axis=0)
        result.append(df)
    return pd.concat(result, axis=0).reset_index(drop=True)


class MixUp:
    def __init__(self, mixup_alpha: float, ratio=Optional[Tuple[float]], official_train: bool = True, BUS: bool = True,
                 USG: bool = True, data_folder_path: str = os.path.join(os.pardir, 'data'), seed: str = 42):
        """
        对图像进行 Mixup 增广并保存。
        划分验证集和训练集时，若图片A在验证集中，则训练集中不能含有任何包含该图片的mixup，当每个样本通过mixup生成s张图片时，
        能够加入训练集的图片个数为sn(1-ratio)E(X)，其中X~B(1;1-ratio)，所以期望为sn(1-ratio)^2，方差为(sn)^2(1-ratio)^3ratio，其中ratio为验证集比例
        若采用k折交叉验证，则ratio=1/k
        :param mixup_alpha: Mixup 中 Beta 分布的参数。这里alpha对应的是认为是真实label的图片所占的比例。
        :param ratio: 各个类别通过增广得到的数量与原来的数量之比，默认都是1，若为小数x，会保证该类中的每张图片增广x下取整次，小数部分随机选取
        :param official_train:
        :param BUS:
        :param USG:
        :param data_folder_path:
        :param seed:
        """
        self.data_folder_path = data_folder_path
        self.table = make_table(data_folder_path=data_folder_path, official_train=official_train, BUS=BUS, USG=USG)
        if not ratio:
            ratio = np.ones(self.table.label.nunique(), dtype=np.int_)
        if not isinstance(ratio, np.ndarray):
            ratio = np.array(ratio)
        self.table = make_ratio_table(self.table, ratio)
        self.lam = np.random.beta(mixup_alpha, mixup_alpha)
        self.fingerprint = f"\nMixup(mixup_alpha={mixup_alpha},official_train={official_train},BUS={BUS},USG={USG}),\n\n"
        self.short_description, self.full_description = make_fingerprint(self, ratio)
        self.dst_folder = os.path.join(self.data_folder_path, 'breast', 'cla', 'augmented', self.short_description)
        self.ratio = ratio

    def __str__(self) -> str:
        return self.fingerprint

    def process_image(self) -> None:
        if os.path.exists(self.dst_folder):
            warn(f"{self.short_description} already exists! stop augment")
            return
        os.makedirs(self.dst_folder)
        self.table['noise_image'] = np.random.randint(0, len(self.table), (len(self.table), 1))
        self.table.noise_image = self.table.noise_image.apply(lambda x: self.table.file_name[x])
        self.table.file_name = self.table.apply(self.mixup, axis=1)
        self.table.drop(['noise_image', 'no'], axis=1, inplace=True)
        self.table.to_csv(os.path.join(self.dst_folder, 'ground_truth.csv'), index=False)
        with open(os.path.join(self.dst_folder, 'README.txt'), 'w') as file:
            file.write(self.full_description)

    def mixup(self, row) -> str:
        """
        apply辅助函数，执行以下功能：
        1. 执行mixup
        2. 形成文件名
        :param row:
        :return:
        """
        origin, noise, no = row.file_name, row.noise_image, row.no
        origin_image, noise_image = cv2.imread(origin), cv2.imread(noise)
        # 调整 Mixup 图像尺寸为原图像尺寸
        noise_image = cv2.resize(noise_image, (origin_image.shape[1], origin_image.shape[0]))
        # 计算 Mixup 权重
        mixup_image = (self.lam * origin_image + (1 - self.lam) * noise_image).astype(np.uint8)
        file_name = origin.split(os.sep)[-1] + "__mixup__" + noise.split(os.sep)[-1] + "(" + str(no) + ').jpg'
        cv2.imwrite(filename=os.path.join(self.dst_folder, file_name), img=mixup_image)
        return file_name


class Preprocess:
    def __init__(self, transform: A.Compose, ratio: Optional[Tuple[float]] = None,
                 data_folder_path: str = os.path.join(os.getcwd(), 'data'), official_train: bool = True, BUS: bool = True,
                 USG: bool = True,fea_official_train:bool=False):
        self.transform = transform
        self.data_folder_path = data_folder_path
        self.table = make_table(data_folder_path=self.data_folder_path, official_train=official_train, BUS=BUS, USG=USG,fea_official_train=fea_official_train)
        if not ratio:
            ratio = np.ones(self.table.label.nunique())
        if not isinstance(ratio, np.ndarray):
            ratio = np.array(ratio)
        if fea_official_train:
            self.task = 'fea'
            ratio = None
        else:
            self.task = "cla"
        
        self.short_description, self.full_description = make_fingerprint(transform, ratio)
        if not fea_official_train:
            self.table = make_ratio_table(self.table, ratio)
        self.dst_folder = os.path.join(self.data_folder_path, 'breast', self.task, 'augmented', self.short_description)
        self.ratio = ratio

    def read_transform_write(self, row)->str:
        """
        apply辅助函数
        :param row:
        :return:
        """
        image_path, no = row.file_name, row.no
        image = cv2.imread(filename=image_path)
        image = np.array(self.transform(image=image)['image'])
        file_name = image_path.split(os.sep)[-1][:-4] + "(" + str(no) + ")" + image_path[-4:]
        cv2.imwrite(os.path.join(self.dst_folder, file_name), image)
        return file_name

    def process_image(self)->None:
        if os.path.exists(self.dst_folder):
            warn(f"{self.short_description} already exists! stop augment")
            return
        os.makedirs(self.dst_folder)
        if self.task == 'cla':
            self.table.file_name = self.table.apply(self.read_transform_write, axis=1)
            self.table.drop(["no"], axis=1, inplace=True)
        else:
            self.table.file_name.apply(lambda x:cv2.imwrite(os.path.join(self.dst_folder,x.split(os.sep)[-1]),np.array(self.transform(image=cv2.imread(filename=x))['image'])))
        self.table.to_csv(os.path.join(self.dst_folder, 'ground_truth.csv'), index=False)
        with open(os.path.join(self.dst_folder, 'README.txt'), 'w') as file:
            file.write(self.full_description)


if __name__ == '__main__':
    transform = A.Compose([A.Rotate(limit=10, always_apply=True)])
    Preprocess(transform,official_train=False,BUS=False,USG=False,fea_official_train=True).process_image()

    ratio = [2, 1, 3, 4, 5, 6]
    # MixUp(0.4, ratio=ratio).process_image()

    # transform = A.Compose([A.Rotate(limit=10, always_apply=True), A.HorizontalFlip(always_apply=True)])
    # Preprocess(transform, ratio=ratio).process_image()

    # transform = A.Compose([A.Rotate(limit=10, always_apply=True)])
    # Preprocess(transform, ratio=ratio).process_image()

    # transform = A.Compose([A.RandomBrightnessContrast(always_apply=True)])
    # Preprocess(transform, ratio=ratio).process_image()

    transform = A.Compose([A.VerticalFlip(always_apply=True)])
    Preprocess(transform, ratio=ratio).process_image()


    # transform = A.Compose([A.Perspective(scale=(0.05, 0.1), always_apply=True)])
    # Preprocess(transform, ratio=ratio).process_image()

    # transform = A.Compose([A.ElasticTransform(alpha=1, sigma=50, always_apply=True)])
    # Preprocess(transform, ratio=ratio).process_image()

