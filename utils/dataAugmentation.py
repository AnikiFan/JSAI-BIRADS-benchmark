import os
import cv2
import numpy as np
import pandas as pd
import albumentations as A
from utils.BreastDataset import make_table
import re
from warnings import warn
from typing import *
from tqdm import tqdm

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

    if type(ratio) != type(None):
        rounded_ratio = [round(r, 1) for r in ratio]
        fingerprint += 'ratio=' + str(tuple(rounded_ratio))
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
    whole = np.floor(ratio).astype(np.int_)
    left = ratio - whole
    result = []
    for label, group in table.groupby('label'):
        df = []
        # 针对每个标签单独检查
        if whole[label] != 0:
            df.append(pd.concat([group.assign(no=int(i)) for i in range(1, whole[label]+1)], axis=0))
        if left[label] != 0:
            df.append(group.iloc[:int(len(group)*left[label]), :].assign(no=int(whole[label]+1)))
        if df:  # 确保df不为空
            result.append(pd.concat(df, axis=0))
    return pd.concat(result, axis=0).reset_index(drop=True)


def print_transformations_info(full_description: str):
    print(f"----------------------processing-----------------------------------\n")
    print(f"{full_description}\n")
    print(f"-------------------------------------------------------------------\n")


def find_next_augmented_folder(data_folder_path: str, short_description: str, full_description: str) -> int:
    """
    查找下一个可用的增强文件夹编号。如果存在相同描述的文件夹，则发出警告并停止增强。

    :param data_folder_path: 数据文件夹的路径。
    :param short_description: 简短描述，用于文件夹命名。
    :param full_description: 完整描述，用于README.txt内容比较。
    :return: 下一个可用的文件夹编号。
    """
    i = 1
    while os.path.exists(os.path.join(data_folder_path, 'breast', 'cla', 'augmented', f"{short_description}-{i}")):
        augmented_path = os.path.join(data_folder_path, 'breast', 'cla', 'augmented', f"{short_description}-{i}")
        readme_path = os.path.join(augmented_path, 'README.txt')
        
        # 检查 README.txt 是否存在
        if not os.path.exists(readme_path):
            warn(f"在 {short_description}-{i} 中未找到 README.txt，请手动检查！")
            return i
        
        # 读取 README.txt 内容并进行比较
        with open(readme_path, 'r') as file:
            if file.read().strip() == full_description.strip():
                warn(f"{full_description} 已存在！停止数据增强。")
                return -1  # 返回-1表示已存在对应描述的文件夹，停止操作
        
        i += 1
    
    return i

class MixUp:
    def __init__(self, mixup_alpha: float, ratio=Optional[Tuple[float]], official_train: bool = True, BUS: bool = True,
                 USG: bool = True, data_folder_path: str = os.path.join(os.getcwd(), 'data'), seed: str = 42):
        """
        对图像进行 Mixup 增广并保存。
        划分验证集和训练集时，若图片A在验证集中，则训练集中不能含有任何包含该图片的mixup，当每个样本通过mixup生成s张图片时，若采用
        k折交叉验证，能够采用为训练集的由mixup额外生成的图片数量的期望为s*n*(k-1)^2/k^2，方差是s*n*(k-1)^2/k^3
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
        self.dst_folder = os.path.join(self.data_folder_path, 'breast', 'cla', 'augmented', f"{self.short_description}-{1}")
        self.ratio = ratio
        tqdm.pandas()  # 初始化 tqdm 的 pandas 支持

    def __str__(self) -> str:
        return self.fingerprint

    def process_image(self) -> None:
        print_transformations_info(self.full_description)
        
        i = find_next_augmented_folder(self.data_folder_path, self.short_description, self.full_description)
        if i == -1:
            return
            
        self.dst_folder = os.path.join(self.data_folder_path, 'breast', 'cla', 'augmented', f"{self.short_description}-{i}")
        
        os.makedirs(self.dst_folder)
        self.table['noise_image'] = np.random.randint(0, len(self.table), (len(self.table), 1))
        self.table.noise_image = self.table.noise_image.apply(lambda x: self.table.file_name[x])
        # 使用 progress_apply 以显示进度条
        self.table.file_name = self.table.progress_apply(self.mixup, axis=1)
        self.table.drop(['noise_image', 'no'], axis=1, inplace=True)
        self.table.to_csv(os.path.join(self.dst_folder, 'ground_truth.csv'), index=False)
        
        # 添加类别数量统计
        category_counts = self.table['label'].value_counts().reset_index()
        category_counts.columns = ['类别', '图片数量']
        category_counts.to_csv(os.path.join(self.dst_folder, 'category_counts.csv'), index=False)
        
        # 打印类别数量信息
        print("数据增强后的类别数量统计：")
        print(category_counts)
        
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
                 USG: bool = True, fea_official_train: bool = False):
        self.transform = transform
        self.data_folder_path = data_folder_path
        self.table = make_table(data_folder_path=self.data_folder_path, official_train=official_train, BUS=BUS, USG=USG, fea_official_train=fea_official_train)
        if not ratio:
            ratio = np.ones(self.table.label.nunique())
        if not isinstance(ratio, np.ndarray):
            ratio = np.array(ratio)
        if fea_official_train:
            self.task = 'fea'
            ratio = None
        else:
            self.task = "cla"
        print(self.table.label.value_counts())
        self.short_description, self.full_description = make_fingerprint(transform, ratio)
        if not fea_official_train:
            self.table = make_ratio_table(self.table, ratio)
        self.dst_folder = os.path.join(self.data_folder_path, 'breast', self.task, 'augmented', f"{self.short_description}-{1}")
        self.ratio = ratio
        tqdm.pandas()  # 初始化 tqdm 的 pandas 支持

    def read_transform_write(self, row) -> str:
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

    def process_image(self) -> None:
        # if os.path.exists(self.dst_folder):
        #     warn(f"{self.short_description} already exists! stop augment")
        #     return
        print_transformations_info(self.full_description)
        
        i = find_next_augmented_folder(self.data_folder_path, self.short_description, self.full_description)
        if i == -1:
            return

        self.dst_folder = os.path.join(self.data_folder_path, 'breast', 'cla', 'augmented', f"{self.short_description}-{i}")
        
        os.makedirs(self.dst_folder)
        if self.task == 'cla':
            self.table.file_name = self.table.progress_apply(self.read_transform_write, axis=1)
            self.table.drop(["no"], axis=1, inplace=True)
        else:
            for x in tqdm(self.table.file_name, desc="处理图像"):
                cv2.imwrite(os.path.join(self.dst_folder, x.split(os.sep)[-1]), np.array(self.transform(image=cv2.imread(filename=x))['image']))
        self.table.to_csv(os.path.join(self.dst_folder, 'ground_truth.csv'), index=False)
        
        # 添加类别数量统计
        category_counts = self.table['label'].value_counts().reset_index()
        category_counts.columns = ['类别', '图片数量']
        category_counts.to_csv(os.path.join(self.dst_folder, 'category_counts.csv'), index=False)
        
        # 打印类别数量信息
        print("数据增强后的类别数量统计：")
        print(category_counts)
        
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


    # transform = A.Compose([A.ElasticTransform(alpha=1, sigma=50, always_apply=True)])
    # Preprocess(transform, ratio=ratio).process_image()