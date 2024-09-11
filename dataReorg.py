# 此文件用于重新整理数据集，将数据集中的图片按照类别分别存放在不同的文件夹中

import sys
print(sys.version)
print(sys.executable)
import pkgutil
print(dir(pkgutil))
import collections
import random
import math
import os
import shutil
import pandas as pd
import torch 
import torchvision
from torch import nn
from d2l import torch as d2l

# 示例路径
dest_dir = r'\超声乳腺影像的BIRADS分类及特征识别\train_valid_test'
train_dir = r'\超声乳腺影像的BIRADS分类及特征识别\超声乳腺训练数据+测试集A+说明文档\乳腺分类训练数据集\train'
test_dir = r'\超声乳腺影像的BIRADS分类及特征识别\超声乳腺训练数据+测试集A+说明文档\test_A\乳腺分类测试集A\A'
valid_ratio = 0.1


def copy_file(src_file, dest_dir):
    """将文件复制到目标目录"""
    os.makedirs(dest_dir, exist_ok=True)
    shutil.copy(src_file, dest_dir)
    


def organize_train_valid(train_dir, dest_dir, valid_ratio):
    """组织训练集和验证集"""
    # 创建目标目录结构
    train_dest_dir = os.path.join(dest_dir, 'train')
    train_valid_dest_dir = os.path.join(dest_dir, 'train_valid')
    valid_dest_dir = os.path.join(dest_dir, 'valid')
    
    os.makedirs(train_dest_dir, exist_ok=True)
    os.makedirs(train_valid_dest_dir, exist_ok=True)
    os.makedirs(valid_dest_dir, exist_ok=True)
    
    for class_name in os.listdir(train_dir):
        class_images_dir = os.path.join(train_dir, class_name, 'images')
        if not os.path.exists(class_images_dir):
            continue
        
        images = os.listdir(class_images_dir)
        num_valid = max(1, math.floor(len(images) * valid_ratio))
        
        # 随机选择部分图像作为验证集
        valid_images = random.sample(images, num_valid)
        
        for image_name in images:
            src_image_path = os.path.join(class_images_dir, image_name)
            # 复制到 train_valid 文件夹
            copy_file(src_image_path, os.path.join(train_valid_dest_dir, class_name))
            # 复制到 valid 或 train 文件夹
            if image_name in valid_images:
                copy_file(src_image_path, os.path.join(valid_dest_dir, class_name))
            else:
                copy_file(src_image_path, os.path.join(train_dest_dir, class_name))
                


def organize_test(test_dir, dest_dir):
    """组织测试集"""
    test_dest_dir = os.path.join(dest_dir, 'test', 'unknown')
    os.makedirs(test_dest_dir, exist_ok=True)
    
    for class_name in os.listdir(test_dir):
        class_images_dir = os.path.join(test_dir, class_name, 'images')
        if not os.path.exists(class_images_dir):
            continue
        
        for image_name in os.listdir(class_images_dir):
            src_image_path = os.path.join(class_images_dir, image_name)
            copy_file(src_image_path, test_dest_dir)


def reorg_data(train_dir, test_dir, dest_dir, valid_ratio=0.1):
    """调用上面的函数，组织训练集、验证集和测试集"""
    organize_train_valid(train_dir, dest_dir, valid_ratio)
    organize_test(test_dir, dest_dir)
    
reorg_data(train_dir, test_dir, dest_dir, valid_ratio=valid_ratio)


