import os
import cv2
import random
import numpy as np
import albumentations as A
from ClaDataset import make_table
import re
from warnings import warn


def make_fingerprint(transform):
    fingerprint = '_'.join(str(transform).splitlines()[1:-1])
    fingerprint = re.sub(r'[<>:"/\\|?*]', '_', fingerprint)
    fingerprint = fingerprint.lower()
    fingerprint = fingerprint.replace(' ','')
    return fingerprint


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
        self.fingerprint = make_fingerprint(self)
        self.fingerprint = f"mixup(mixup_alpha={mixup_alpha},official_train={official_train},BUS={BUS},USG={USG})"
        self.dst_folder = os.path.join(self.data_folder_path, 'breast', 'cla', 'augmented', self.fingerprint)

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
        noise_image = cv2.resize(noise_image, (origin_image.shape[1], origin_image.shape[0]))
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
        self.dst_folder = os.path.join(self.data_folder_path, 'breast', 'cla', 'augmented', self.fingerprint)

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


if __name__ == '__main__':
    MixUp(0.4).process_image()

    transform = A.Compose([A.Rotate(limit=10, p=1)])
    Preprocess(transform, make_fingerprint(transform)).process_image()

    transform = A.Compose([A.RandomBrightnessContrast(p=1)])
    Preprocess(transform, make_fingerprint(transform)).process_image()

    transform = A.Compose([A.Perspective(scale=(0.05, 0.1), p=1)])
    Preprocess(transform, make_fingerprint(transform)).process_image()

    transform = A.Compose([A.ElasticTransform(alpha=1, sigma=50, p=1)])
    Preprocess(transform, make_fingerprint(transform)).process_image()
