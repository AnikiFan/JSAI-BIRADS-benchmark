import os
import cv2
import random
import numpy as np
import albumentations as A
from ClaDataset import make_table
import re
from warnings import warn


def make_fingerprint(transform_name, params):
    transform_name = re.sub(r'[<>:"/\\|?*]', '_', transform_name)
    transform_name = transform_name.lower()
    # 处理参数，将其转换为字符串形式
    param_str = '_'.join(f"{key}={value}" for key, value in params.items())
    param_str = re.sub(r'[<>:"/\\|?*]', '_', param_str)
    return transform_name + '_' + param_str


class MixUp:
    def __init__(self, mixup_alpha, official_train=True, BUS=True, USG=True,
                 data_folder_path=os.path.join(os.pardir, 'data'), seed=42):
        """
        对图像进行 Mixup 增广并保存。
        划分验证集和训练集时，若图片A在验证集中，则训练集中不能含有任何包含该图片的mixup，当每个样本通过mixup生成s张图片时，若采用
        k折交叉验证，能够采用为训练集的由mixup额外生成的图片数量的期望为s*n*(k-1)^2/k^2

        Args:
            mixup_alpha (float): Mixup 中 Beta 分布的参数。这里alpha对应的是认为是真实label的图片所占的比例。
        """
        self.data_folder_path = data_folder_path
        self.table = make_table(data_folder_path=data_folder_path, official_train=official_train, BUS=BUS, USG=USG)
        self.lam = np.random.beta(mixup_alpha, mixup_alpha)
        self.fingerprint = make_fingerprint('Mixup',
                                            {"mixup_alpha": mixup_alpha, "official_train": official_train, "BUS": BUS,
                                             "USG": USG})
        self.dst_folder = os.path.join(self.data_folder_path, 'breast', 'cla', self.fingerprint)

    def process_image(self):
        if os.path.exists(self.dst_folder):
            warn(f"{self.fingerprint} already exists! stop augment")
            return
        os.makedirs(self.dst_folder)
        self.table['noise_image'] = np.random.randint(0, len(self.table), (len(self.table), 1))
        self.table.noise_image = self.table.noise_image.apply(lambda x: self.table.loc[x, 'file_name'])
        self.table.file_name += '__mixup__'
        self.table.file_name += self.table.noise_image
        self.table.drop(['noise_image'], axis=1, inplace=True)
        self.table.file_name = self.table.file_name.str.split('__mixup__').apply(self.mixup)
        self.table.to_csv(os.path.join(self.dst_folder, 'ground_truth.csv'), index=False)

    def mixup(self, pair):
        origin, noise = pair
        origin_image, noise_image = cv2.imread(origin), cv2.imread(noise)
        # 调整 Mixup 图像尺寸为原图像尺寸
        noise_image = cv2.resize(noise_image, (origin_image.shape[1],origin_image.shape[0]))
        # 计算 Mixup 权重
        mixup_image = (self.lam * origin_image + (1 - self.lam) * noise_image).astype(np.uint8)
        file_name = origin.split(os.sep)[-1] + "__mixup__" + noise.split(os.sep)[-1] + '.jpg'
        cv2.imwrite(filename=os.path.join(self.dst_folder, file_name), img=mixup_image)
        return file_name


class Preprocess:
    def __init__(self, transform, fingerprint, data_folder_path=os.path.join(os.pardir, 'data')):
        self.transform = transform
        self.fingerprint = fingerprint
        self.data_folder_path = data_folder_path
        self.dst_folder = os.path.join(self.data_folder_path, 'breast', 'cla', self.fingerprint)

    def read_transform_write(self, image_path, dst_folder):
        image = cv2.imread(filename=image_path)
        image = np.array(self.transform(image=image)['image'])
        cv2.imwrite(os.path.join(self.dst_folder, image_path.split(os.sep)[-1]), image)

    def process_image(self):
        if os.path.exists(self.dst_folder):
            warn(f"{self.fingerprint} already exists! stop augment")
            return
        os.makedirs(self.dst_folder)
        table = make_table(data_folder_path=self.data_folder_path, official_train=True, BUS=True, USG=True)
        table.file_name.apply(lambda file_name: self.read_transform_write(file_name, self.dst_folder))
        table.file_name = table.file_name.str.split(os.sep).apply(lambda x: x[-1])
        table.to_csv(os.path.join(self.dst_folder, 'ground_truth.csv'), index=False)


def main():
    # 源数据集路径（原始数据集）
    src_root = '/Users/huiyangzheng/Desktop/Project/Competition/GCAIAEC2024/AIC/TDS-Net/data/乳腺分类训练数据集/train_split_0.9'

    # 目标数据集路径（增广后的数据集）
    dst_root = '/Users/huiyangzheng/Desktop/Project/Competition/GCAIAEC2024/AIC/TDS-Net/data/乳腺分类训练数据集/train_split_0.9_augmented_demo'

    # 原数据集各类别数量：
    # 2类: 463
    # 3类: 878
    # 4A类: 448
    # 4B类: 295
    # 4C类: 251
    # 5类: 138
    # 定义每个类别的增广次数，针对少数类别进行更多增广
    class_aug_times = {
        '2类': 2,  # 463
        '3类': 1,  # 878
        '4A类': 3,  # 448
        '4B类': 4,  # 295
        '4C类': 5,  # 251
        '5类': 6  # 138
    }

    # 定义增广策略列表，包含尺寸调整和多种增广方法
    augmentations = [
        # 尺寸调整
        A.Compose([
            A.Resize(height=224, width=224, p=1.0),
        ]),

        # 水平翻转
        A.Compose([
            A.HorizontalFlip(p=1.0),
        ]),

        # 垂直翻转
        # A.Compose([
        #     A.VerticalFlip(p=1.0),
        # ]),

        # 随机旋转一定角度
        A.Compose([
            A.Rotate(limit=10, p=1.0),
        ]),

        # 随机亮度和对比度调整
        A.Compose([
            A.RandomBrightnessContrast(p=1.0),
        ]),

        # # 高斯噪声
        # A.Compose([
        #     A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
        # ]),

        # 仿射变换
        # A.Compose([
        #     A.Affine(scale=(0.9, 1.1), translate_percent=(0.1, 0.1), rotate=(-15, 15), shear=(-10, 10), p=1.0),
        # ]),

        # 颜色抖动，乳腺超声影像为灰度图，颜色抖动对图像无实际意义，建议移除。
        # A.Compose([
        #     A.ColorJitter(p=1.0),
        # ]),

        # 随机擦除
        # A.Compose([
        #     A.CoarseDropout(max_holes=8, max_height=16, max_width=16, fill_value=0, p=1.0),
        # ]),

        # 透视变换
        A.Compose([
            A.Perspective(scale=(0.05, 0.1), p=1.0),
        ]),

        # 弹性变换
        A.Compose([
            A.ElasticTransform(alpha=1.0, sigma=50.0, p=1.0),
        ]),
    ]

    # 启用 Mixup，并设置 alpha 参数
    use_mixup = True
    mixup_alpha = 0.4  # Beta 分布的参数，控制混合比例

    create_augmented_dataset(src_root, dst_root, class_aug_times, augmentations, use_mixup, mixup_alpha)


if __name__ == '__main__':
    MixUp(0.4).process_image()

    transform_name = 'Rotate'
    params = {"limit": 10, "p": 1.0}
    Preprocess(getattr(A, transform_name)(**params), make_fingerprint(transform_name, params)).process_image()

    transform_name = 'RandomBrightnessContrast'
    params = {"p": 1.0}
    Preprocess(getattr(A, transform_name)(**params), make_fingerprint(transform_name, params)).process_image()

    transform_name = 'Perspective'
    params = {"scale": (0.05, 0.1), "p": 1.0}
    Preprocess(getattr(A, transform_name)(**params), make_fingerprint(transform_name, params)).process_image()

    transform_name = 'ElasticTransform'
    params = {"alpha":1.0,"sigma":50.0, "p": 1.0}
    Preprocess(getattr(A, transform_name)(**params), make_fingerprint(transform_name, params)).process_image()


