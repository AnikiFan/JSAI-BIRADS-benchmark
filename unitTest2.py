
import random
import unittest
import pandas as pd
import numpy as np
import os

import torch
from PIL.Image import Image
from torch import Tensor
from numpy import ndarray
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip
from utils.BreastDataset import make_table, getBreastTrainValidData, in_valid, split_augmented_image, \
    BreastCrossValidationData
from utils.TableDataset import TableDataset
from unittest.mock import mock_open, patch
from utils.loss_function import MyBCELoss
from utils.metrics import my_multilabel_accuracy, multilabel_f1_score, multilabel_confusion_matrix
from utils.dataAugmentation import Preprocess,MixUp
from glob import glob
import albumentations as A


class GroundTruthConsistencyTestCase(unittest.TestCase):
    def setUp(self):
        # 读取 ground_truth.csv 文件
        self.ground_truth_path = os.path.join('data', 'breast', 'cla', 'train', 'ground_truth.csv')
        self.ground_truth = pd.read_csv(self.ground_truth_path)

        # 获取 official_train 中的类别目录
        self.official_train_dir = '/mnt/AIC/DLApproach/data/breast/cla/official_train'
        self.class_dirs = [d for d in os.listdir(self.official_train_dir) if os.path.isdir(os.path.join(self.official_train_dir, d))]

        # 创建文件名到实际类别的映射
        class_dict = {
            '2类':0,
            '3类':1,
            '4A类':2,
            '4B类':3,
            '4C类':4,
            '5类':5
        }
        self.file_to_actual_class = {}
        for class_name in self.class_dirs:
            class_dir = os.path.join(self.official_train_dir, class_name,"images")
            for root, _, files in os.walk(class_dir):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.file_to_actual_class[file] = class_dict[class_name]

    def test_classification_consistency(self):
        mismatches = []
        for _, row in self.ground_truth.iterrows():
            file_name = row['file_name']
            expected_label = int(row['label'])
            # expected_label = self.class_dict[expected_label]

            actual_class = self.file_to_actual_class.get(file_name)
            if actual_class is None:
                mismatches.append((file_name, expected_label, '文件未找到'))
            elif actual_class != expected_label:
                mismatches.append((file_name, expected_label, actual_class))

        # 如果存在不匹配，打印详细信息
        if mismatches:
            for mismatch in mismatches:
                print(f"文件名: {mismatch[0]}, 预期类别: {mismatch[1]} | 实际类别: {mismatch[2]}")
        self.assertEqual(len(mismatches), 0, f"存在 {len(mismatches)} 个文件的分类不一致")



if __name__ == '__main__':
    unittest.main()
