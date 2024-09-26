import random
import math
import os
import shutil
from tqdm import tqdm


class DataOrganizer:
    def __init__(self, dest_dir, train_dir, test_dir,valid_ratio):
        self.dest_dir = dest_dir
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.valid_ratio = valid_ratio

    def copy_file(self,src_file,dest_dir):
        """将文件复制到目标目录"""
        os.makedirs(self.dest_dir, exist_ok=True)
        shutil.copy(src_file, dest_dir)

    def organize_train_valid(self):
        """组织训练集和验证集"""
        # 创建目标目录结构
        train_dest_dir = os.path.join(self.dest_dir, 'train')
        train_valid_dest_dir = os.path.join(self.dest_dir, 'train_valid')
        valid_dest_dir = os.path.join(self.dest_dir, 'valid')

        os.makedirs(self.dest_dir, exist_ok=True)
        os.makedirs(self.dest_dir, exist_ok=True)
        os.makedirs(self.dest_dir, exist_ok=True)

        for class_name in tqdm(os.listdir(self.train_dir)):
            class_images_dir = os.path.join(self.train_dir, class_name, 'images')
            if not os.path.exists(class_images_dir):
                continue

            images = os.listdir(class_images_dir)
            num_valid = max(1, math.floor(len(images) * self.valid_ratio))

            # 随机选择部分图像作为验证集
            valid_images = random.sample(images, num_valid)

            for image_name in tqdm(images):
                src_image_path = os.path.join(class_images_dir, image_name)
                # 复制到 train_valid 文件夹
                self.copy_file(src_image_path, os.path.join(train_valid_dest_dir, class_name))
                # 复制到 valid 或 train 文件夹
                if image_name in valid_images:
                    self.copy_file(src_image_path, os.path.join(valid_dest_dir, class_name))
                else:
                    self.copy_file(src_image_path, os.path.join(train_dest_dir, class_name))

    def organize_test(self):
        """组织测试集"""
        test_dest_dir = os.path.join(self.dest_dir, 'test', 'unknown')
        os.makedirs(test_dest_dir, exist_ok=True)

        for class_name in tqdm(os.listdir(self.test_dir)):
            class_images_dir = os.path.join(self.test_dir, class_name, 'images')
            if not os.path.exists(class_images_dir):
                continue

            for image_name in tqdm(os.listdir(class_images_dir)):
                src_image_path = os.path.join(class_images_dir, image_name)
                self.copy_file(src_image_path, test_dest_dir)

    def reorg_data(self, valid_ratio=0.1):
        """调用上面的函数，组织训练集、验证集和测试集"""
        self.organize_train_valid()
        self.organize_test()


if __name__ == '__main__':
    dest_dir = os.path.join(os.getcwd(), "data", "breast", "train_valid_test")
    train_dir = os.path.join(os.getcwd(), "data", "breast", "train", "cla")
    test_dir = os.path.join(os.getcwd(), "data", "breast", "test_A", "cla")
    organizer = DataOrganizer(dest_dir=dest_dir,test_dir=test_dir, train_dir=train_dir, valid_ratio=0.1)
    organizer.reorg_data()
