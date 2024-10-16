import random
import unittest
import pandas as pd
import numpy as np
import os
import re
import torch
from PIL.Image import Image
from numpy.ma.testutils import assert_equal
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

class FolderTestCase(unittest.TestCase):
    def test_official_file(self):
        """
        测试是否有官方数据文件夹
        :return:
        """
        self.assertTrue(os.path.exists(os.path.join(os.curdir,'data','breast','cla','official_train')))
        self.assertTrue(os.path.exists(os.path.join(os.curdir,'data','breast','cla','official_test')))
        self.assertTrue(os.path.exists(os.path.join(os.curdir, 'data', 'breast', 'fea', 'official_train')))
        self.assertTrue(os.path.exists(os.path.join(os.curdir, 'data', 'breast', 'fea', 'official_test')))
        self.assertTrue(os.path.exists(os.path.join(os.curdir,'docs','cla_order.csv')))
        self.assertTrue(os.path.exists(os.path.join(os.curdir,'docs','fea_order.csv')))

    def test_official_file_tree(self):
        self.assertCountEqual(['2类','3类','4A类','4B类','4C类','5类'],os.listdir(os.path.join(os.curdir,'data','breast','cla','official_train')))
        self.assertCountEqual(['2类','3类','4A类','4B类','4C类','5类'],os.listdir(os.path.join(os.curdir,'data','breast','cla','official_test')))
        self.assertCountEqual(['boundary_labels','calcification_labels','direction_labels','images','shape_labels'],os.listdir(os.path.join(os.curdir,'data','breast','fea','official_train')))
        self.assertCountEqual(['boundary_labels','calcification_labels','direction_labels','images','shape_labels'],os.listdir(os.path.join(os.curdir,'data','breast','fea','official_train')))

    def test_organized_file(self):
        """
        测试是否运行过OfficialDataOrganize.py
        :return:
        """
        self.assertTrue(os.path.exists(os.path.join(os.curdir, 'data', 'breast', 'cla', 'train')))
        self.assertTrue(os.path.exists(os.path.join(os.curdir, 'data', 'breast', 'cla', 'test')))
        self.assertTrue(os.path.exists(os.path.join(os.curdir, 'data', 'breast', 'fea', 'train')))
        self.assertTrue(os.path.exists(os.path.join(os.curdir, 'data', 'breast', 'fea', 'test')))
        self.assertTrue(os.path.exists(os.path.join(os.curdir, 'data', 'breast', 'cla', 'train','ground_truth.csv')))
        self.assertTrue(os.path.exists(os.path.join(os.curdir, 'data', 'breast', 'cla', 'test','ground_truth.csv')))
        self.assertTrue(os.path.exists(os.path.join(os.curdir, 'data', 'breast', 'fea', 'train','ground_truth.csv')))
        self.assertTrue(os.path.exists(os.path.join(os.curdir, 'data', 'breast', 'fea', 'test','ground_truth.csv')))


class CrossValidationTestCase(unittest.TestCase):
    data_folder_path = os.path.join(os.curdir, 'data')
    k_fold = 5
    if not len(glob(os.path.join(os.curdir, 'data','breast', 'cla','augmented','Mixup,ratio=(1.0,1.0,1.0,1.0,1.0,1.0)-*'))):
        MixUp(0.2,[1,1,1,1,1,1]).process_image()
    if not len(glob(os.path.join(os.curdir, 'data', 'breast', 'cla', 'augmented', 'Mixup,ratio=(2.0,1.0,3.0,4.0,5.0,6.0)-*'))):
        MixUp(0.2, [2,1,3,4,5,6]).process_image()
    if not len(glob(os.path.join(os.curdir, 'data', 'breast', 'cla', 'augmented','Rotate,ratio=(1.0,1.0,1.0,1.0,1.0,1.0)-*'))):
        Preprocess(A.Compose([A.Rotate(limit=15,p=1.0)]), [1, 1, 1, 1, 1, 1]).process_image()
    if not len(glob(os.path.join(os.curdir, 'data', 'breast', 'cla', 'augmented','Rotate,ratio=(2.0,1.0,3.0,4.0,5.0,6.0)-*'))):
        Preprocess(A.Compose([A.Rotate(limit=15,p=1.0)]), [2, 1, 3, 4, 5, 6]).process_image()
    if not len(glob(os.path.join(os.curdir, 'data', 'breast', 'fea', 'augmented','Mixup-*'))):
        MixUp(0.2, official_train=False,BUS=False,USG=False,fea_official_train=True).process_image()
    if not len(glob(os.path.join(os.curdir, 'data', 'breast', 'fea', 'augmented', 'VerticalFlip-*'))):
        Preprocess(A.Compose([A.VerticalFlip(p=1)]), official_train=False, BUS=False, USG=False, fea_official_train=True).process_image()

    cla_mixup_folder1 = glob(os.path.join(os.curdir, 'data','breast', 'cla','augmented','Mixup,ratio=(1.0,1.0,1.0,1.0,1.0,1.0)-*'))[0]
    cla_mixup_folder2 = glob(os.path.join(os.curdir, 'data','breast', 'cla','augmented','Mixup,ratio=(2.0,1.0,3.0,4.0,5.0,6.0)-*'))[0]
    cla_rotate_folder1 = glob(os.path.join(os.curdir, 'data','breast', 'cla','augmented','Rotate,ratio=(1.0,1.0,1.0,1.0,1.0,1.0)-*'))[0]
    cla_rotate_folder2 = glob(os.path.join(os.curdir, 'data','breast', 'cla','augmented','Rotate,ratio=(2.0,1.0,3.0,4.0,5.0,6.0)-*'))[0]
    fea_mixup_folder = glob(os.path.join(os.curdir, 'data', 'breast', 'fea', 'augmented','Mixup-*'))[0]
    fea_flip_folder = glob(os.path.join(os.curdir, 'data', 'breast', 'fea', 'augmented','VerticalFlip-*'))[0]

    def test_len_without_augment(self):
        """
        不引入数据增广数据集的情况下，测试返回数据集的长度
        :return:
        """
        cla_cross_validation_dataset = BreastCrossValidationData(data_folder_path=self.data_folder_path,
                                                                 k_fold=self.k_fold)
        for train_dataset, valid_dataset in cla_cross_validation_dataset:
            self.assertAlmostEqual(1 / self.k_fold, len(valid_dataset) / (len(valid_dataset) + len(train_dataset)),
                                   delta=0.01)

    def test_leakage_without_augment(self):
        """
        不引入数据增广数据集的情况下，测试是否发生信息泄露
        :return:
        """
        official_train, BUS, USG, fea_official_train = True, True, True, False
        cla_cross_validation_dataset = BreastCrossValidationData(data_folder_path=self.data_folder_path,
                                                                 k_fold=self.k_fold, official_train=official_train,
                                                                 BUS=BUS, USG=USG,
                                                                 fea_official_train=fea_official_train)
        for train_dataset, valid_dataset in cla_cross_validation_dataset:
            valid_list = valid_dataset.table.file_name.str.split(os.sep).str[-1].str[:-4].tolist()
            self.assertFalse(np.any(train_dataset.table.file_name.apply(lambda x:re.sub(r"\(\d\)","",x.split(os.sep)[-1][:-4])).apply(lambda x: any([valid == x for valid in valid_list]))))

        official_train, BUS, USG, fea_official_train = True, False, False, False
        cla_cross_validation_dataset = BreastCrossValidationData(data_folder_path=self.data_folder_path,
                                                                 k_fold=self.k_fold, official_train=official_train,
                                                                 BUS=BUS, USG=USG,
                                                                 fea_official_train=fea_official_train)
        for train_dataset, valid_dataset in cla_cross_validation_dataset:
            valid_list = valid_dataset.table.file_name.str.split(os.sep).str[-1].str[:-4].tolist()
            self.assertFalse(np.any(train_dataset.table.file_name.apply(lambda x:re.sub(r"\(\d\)","",x.split(os.sep)[-1][:-4])).apply(lambda x: any([valid == x for valid in valid_list]))))

        official_train, BUS, USG, fea_official_train = False, True, False, False
        cla_cross_validation_dataset = BreastCrossValidationData(data_folder_path=self.data_folder_path,
                                                                 k_fold=self.k_fold, official_train=official_train,
                                                                 BUS=BUS, USG=USG,
                                                                 fea_official_train=fea_official_train)
        for train_dataset, valid_dataset in cla_cross_validation_dataset:
            valid_list = valid_dataset.table.file_name.str.split(os.sep).str[-1].str[:-4].tolist()
            self.assertFalse(np.any(train_dataset.table.file_name.apply(lambda x:re.sub(r"\(\d\)","",x.split(os.sep)[-1][:-4])).apply(lambda x: any([valid == x for valid in valid_list]))))

        official_train, BUS, USG, fea_official_train = False, False, True, False
        cla_cross_validation_dataset = BreastCrossValidationData(data_folder_path=self.data_folder_path,
                                                                 k_fold=self.k_fold, official_train=official_train,
                                                                 BUS=BUS, USG=USG,
                                                                 fea_official_train=fea_official_train)
        for train_dataset, valid_dataset in cla_cross_validation_dataset:
            valid_list = valid_dataset.table.file_name.str.split(os.sep).str[-1].str[:-4].tolist()
            self.assertFalse(np.any(train_dataset.table.file_name.apply(lambda x:re.sub(r"\(\d\)","",x.split(os.sep)[-1][:-4])).apply(lambda x: any([valid == x for valid in valid_list]))))

        official_train, BUS, USG, fea_official_train = False, False, False, True
        cla_cross_validation_dataset = BreastCrossValidationData(data_folder_path=self.data_folder_path,
                                                                 k_fold=self.k_fold, official_train=official_train,
                                                                 BUS=BUS, USG=USG,
                                                                 fea_official_train=fea_official_train)
        for train_dataset, valid_dataset in cla_cross_validation_dataset:
            valid_list = valid_dataset.table.file_name.str.split(os.sep).str[-1].str[:-4].tolist()
            self.assertFalse(np.any(train_dataset.table.file_name.apply(lambda x:re.sub(r"\(\d\)","",x.split(os.sep)[-1][:-4])).apply(lambda x: any([valid == x for valid in valid_list]))))

    def test_overlap_without_augment(self):
        """
        不引入数据增广数据集的情况下，测试各折的验证集之间是否存在重叠的情况
        :return:
        """
        official_train, BUS, USG, fea_official_train = True, True, True, False
        cla_cross_validation_dataset = BreastCrossValidationData(data_folder_path=self.data_folder_path,
                                                                 k_fold=self.k_fold, official_train=official_train,
                                                                 BUS=BUS, USG=USG,
                                                                 fea_official_train=fea_official_train)
        first = True
        for train_dataset, valid_dataset in cla_cross_validation_dataset:
            if first:
                valid_datasets = valid_dataset.table.file_name
                first = False
                continue
            valid_datasets = np.intersect1d(valid_datasets, valid_dataset.table.file_name)
        self.assertEqual(0, len(valid_datasets))

        official_train, BUS, USG, fea_official_train = True, False, False, False
        cla_cross_validation_dataset = BreastCrossValidationData(data_folder_path=self.data_folder_path,
                                                                 k_fold=self.k_fold, official_train=official_train,
                                                                 BUS=BUS, USG=USG,
                                                                 fea_official_train=fea_official_train)
        first = True
        for train_dataset, valid_dataset in cla_cross_validation_dataset:
            if first:
                valid_datasets = valid_dataset.table.file_name
                first = False
                continue
            valid_datasets = np.intersect1d(valid_datasets, valid_dataset.table.file_name)
        self.assertEqual(0, len(valid_datasets))

        official_train, BUS, USG, fea_official_train = False, True, False, False
        cla_cross_validation_dataset = BreastCrossValidationData(data_folder_path=self.data_folder_path,
                                                                 k_fold=self.k_fold, official_train=official_train,
                                                                 BUS=BUS, USG=USG,
                                                                 fea_official_train=fea_official_train)
        first = True
        for train_dataset, valid_dataset in cla_cross_validation_dataset:
            if first:
                valid_datasets = valid_dataset.table.file_name
                first = False
                continue
            valid_datasets = np.intersect1d(valid_datasets, valid_dataset.table.file_name)
        self.assertEqual(0, len(valid_datasets))

        official_train, BUS, USG, fea_official_train = False, False, True, False
        cla_cross_validation_dataset = BreastCrossValidationData(data_folder_path=self.data_folder_path,
                                                                 k_fold=self.k_fold, official_train=official_train,
                                                                 BUS=BUS, USG=USG,
                                                                 fea_official_train=fea_official_train)
        first = True
        for train_dataset, valid_dataset in cla_cross_validation_dataset:
            if first:
                valid_datasets = valid_dataset.table.file_name
                first = False
                continue
            valid_datasets = np.intersect1d(valid_datasets, valid_dataset.table.file_name)
        self.assertEqual(0, len(valid_datasets))

        official_train, BUS, USG, fea_official_train = False, False, False, True
        cla_cross_validation_dataset = BreastCrossValidationData(data_folder_path=self.data_folder_path,
                                                                 k_fold=self.k_fold, official_train=official_train,
                                                                 BUS=BUS, USG=USG,
                                                                 fea_official_train=fea_official_train)
        first = True
        for train_dataset, valid_dataset in cla_cross_validation_dataset:
            if first:
                valid_datasets = valid_dataset.table.file_name
                first = False
                continue
            valid_datasets = np.intersect1d(valid_datasets, valid_dataset.table.file_name)
        self.assertEqual(0, len(valid_datasets))

    def test_len_with_mixup_augment(self):
        """
        引入数据增广数据集的情况下，测试返回数据集的长度
        k折交叉验证，能够采用为训练集的由mixup额外生成的图片数量的期望为s*n*(k-1)^2/k^2，方差是s*n*(k-1)^2/k^3
        要求实际能够加入训练集的
        :return:
        """
        cla_cross_validation_dataset_with_augment = BreastCrossValidationData(data_folder_path=self.data_folder_path,
                                                                              k_fold=self.k_fold,
                                                                              augmented_folder_list=[self.cla_mixup_folder1])
        cla_cross_validation_dataset_without_augment = BreastCrossValidationData(data_folder_path=self.data_folder_path,
                                                                                 k_fold=self.k_fold)
        for train_dataset_with_augment, valid_dataset_with_augment in cla_cross_validation_dataset_with_augment:
            train_dataset_without_augment, valid_dataset_without_augment = next(
                cla_cross_validation_dataset_without_augment)
            self.assertGreater(len(train_dataset_with_augment) - len(train_dataset_without_augment),
                               (0.64 - 0.08 * 3) * len(train_dataset_without_augment))
            self.assertLess(len(train_dataset_with_augment) - len(train_dataset_without_augment),
                            (0.64 + 0.08 * 3) * len(train_dataset_without_augment))

    def test_leakage_with_mixup_augment(self):
        """
        引入数据增广数据集的情况下，测试是否发生信息泄露
        :return:
        """
        official_train, BUS, USG, fea_official_train = True, True, True, False
        cla_cross_validation_dataset_with_augmented = BreastCrossValidationData(data_folder_path=self.data_folder_path,
                                                                                k_fold=self.k_fold,
                                                                                official_train=official_train, BUS=BUS,
                                                                                USG=USG,
                                                                                fea_official_train=fea_official_train,
                                                                                augmented_folder_list=[self.cla_mixup_folder2])
        for train_dataset, valid_dataset in cla_cross_validation_dataset_with_augmented:
            valid_list = valid_dataset.table.file_name.str.split(os.sep).str[-1].str[:-4].tolist()
            self.assertFalse(np.any(train_dataset.table.file_name.apply(lambda x:list(map(lambda x:x.replace('.png','').replace('.jpg',''),re.sub(r"\(\d\)","",x.split(os.sep)[-1][:-4]).split('__mixup__')))).apply(lambda x: any([valid in x for valid in valid_list]))))

        official_train, BUS, USG, fea_official_train = False, False, False, True
        fea_cross_validation_dataset_with_augmented = BreastCrossValidationData(data_folder_path=self.data_folder_path,
                                                                                k_fold=self.k_fold,
                                                                                official_train=official_train, BUS=BUS,
                                                                                USG=USG,
                                                                                fea_official_train=fea_official_train,
                                                                                augmented_folder_list=[self.fea_mixup_folder])
        for train_dataset, valid_dataset in fea_cross_validation_dataset_with_augmented:
            valid_list = valid_dataset.table.file_name.str.split(os.sep).str[-1].str[:-4].tolist()
            self.assertFalse(np.any(train_dataset.table.file_name.apply(lambda x:list(map(lambda x:x.replace('.png','').replace('.jpg',''),re.sub(r"\(\d\)","",x.split(os.sep)[-1][:-4]).split('__mixup__')))).apply(lambda x: any([valid in x for valid in valid_list]))))

    def test_overlap_with_mixup_augment(self):
        """
        引入数据增广数据集的情况下，测试各折的验证集之间是否存在重叠的情况
        :return:
        """
        official_train, BUS, USG, fea_official_train = True, True, True, False
        cla_cross_validation_dataset = BreastCrossValidationData(data_folder_path=self.data_folder_path,
                                                                 k_fold=self.k_fold,
                                                                 official_train=official_train, BUS=BUS,
                                                                 USG=USG,
                                                                 fea_official_train=fea_official_train,
                                                                 augmented_folder_list=[self.cla_rotate_folder2])
        first = True
        for train_dataset, valid_dataset in cla_cross_validation_dataset:
            if first:
                valid_datasets = valid_dataset.table.file_name
                first = False
                continue
            valid_datasets = np.intersect1d(valid_datasets, valid_dataset.table.file_name)
        self.assertEqual(0, len(valid_datasets))

        official_train, BUS, USG, fea_official_train = False, False, False, True
        cla_cross_validation_dataset = BreastCrossValidationData(data_folder_path=self.data_folder_path,
                                                                 k_fold=self.k_fold,
                                                                 official_train=official_train, BUS=BUS,
                                                                 USG=USG,
                                                                 fea_official_train=fea_official_train,
                                                                 augmented_folder_list=[self.cla_mixup_folder2])
        first = True
        for train_dataset, valid_dataset in cla_cross_validation_dataset:
            if first:
                valid_datasets = valid_dataset.table.file_name
                first = False
                continue
            valid_datasets = np.intersect1d(valid_datasets, valid_dataset.table.file_name)
        self.assertEqual(0, len(valid_datasets))

    def test_leakage_with_normal_augment(self):
        """
        引入数据增广数据集的情况下，测试是否发生信息泄露
        :return:
        """
        official_train, BUS, USG, fea_official_train = True, True, True, False
        cla_cross_validation_dataset_with_augmented = BreastCrossValidationData(data_folder_path=self.data_folder_path,
                                                                                k_fold=self.k_fold,
                                                                                official_train=official_train, BUS=BUS,
                                                                                USG=USG,
                                                                                fea_official_train=fea_official_train,
                                                                                augmented_folder_list=[self.cla_rotate_folder1])
        for train_dataset, valid_dataset in cla_cross_validation_dataset_with_augmented:
            valid_list = valid_dataset.table.file_name.str.split(os.sep).str[-1].str[:-4].tolist()
            self.assertFalse(np.any(train_dataset.table.file_name.apply(lambda x:re.sub(r"\(\d\)","",x.split(os.sep)[-1][:-4])).apply(lambda x: any([valid == x for valid in valid_list]))))

        official_train, BUS, USG, fea_official_train = False, False, False, True
        fea_cross_validation_dataset_with_augmented = BreastCrossValidationData(data_folder_path=self.data_folder_path,
                                                                                k_fold=self.k_fold,
                                                                                official_train=official_train, BUS=BUS,
                                                                                USG=USG,
                                                                                fea_official_train=fea_official_train,
                                                                                augmented_folder_list=[self.fea_flip_folder])
        for train_dataset, valid_dataset in fea_cross_validation_dataset_with_augmented:
            valid_list = valid_dataset.table.file_name.str.split(os.sep).str[-1].str[:-4].tolist()
            self.assertFalse(np.any(train_dataset.table.file_name.apply(lambda x:re.sub(r"\(\d\)","",x.split(os.sep)[-1][:-4])).apply(lambda x: any([valid == x for valid in valid_list]))))


    def test_len_with_normal_augment(self):
        """
        引入数据增广数据集的情况下，测试返回数据集的长度
        k折交叉验证，能够采用为训练集的由mixup额外生成的图片数量的期望为s*n*(k-1)^2/k^2，方差是s*n*(k-1)^2/k^3
        要求实际能够加入训练集的
        :return:
        """
        cla_cross_validation_dataset_with_augment = BreastCrossValidationData(data_folder_path=self.data_folder_path,
                                                                              k_fold=self.k_fold,
                                                                              augmented_folder_list=[self.cla_rotate_folder1])
        cla_cross_validation_dataset_without_augment = BreastCrossValidationData(data_folder_path=self.data_folder_path,
                                                                                 k_fold=self.k_fold)
        for train_dataset_with_augment, valid_dataset_with_augment in cla_cross_validation_dataset_with_augment:
            train_dataset_without_augment, valid_dataset_without_augment = next(
                cla_cross_validation_dataset_without_augment)
            self.assertAlmostEqual(len(train_dataset_with_augment) - len(train_dataset_without_augment),
                                   len(train_dataset_without_augment))


    def test_overlap_with_normal_augment(self):
        """
        引入数据增广数据集的情况下，测试各折的验证集之间是否存在重叠的情况
        :return:
        """
        cla_cross_validation_dataset = BreastCrossValidationData(data_folder_path=self.data_folder_path,
                                                                 k_fold=self.k_fold,
                                                                 augmented_folder_list=[self.cla_rotate_folder2])
        first = True
        for train_dataset, valid_dataset in cla_cross_validation_dataset:
            if first:
                valid_datasets = valid_dataset.table.file_name
                first = False
                continue
            valid_datasets = np.intersect1d(valid_datasets, valid_dataset.table.file_name)
        self.assertEqual(0, len(valid_datasets))


class ClaInValidTestCase(unittest.TestCase):
    valid_list = [
        "0086.jpg",
        "0087.jpg",
        "0088.jpg",
        "hcjz_birads3_0056_0004.jpg",
        "hcjz_birads3_01170_0001.jpg",
        "hcjz_birads5_0002_0006.jpg",
        "sec_0035.jpg",
        "sec_0036.jpg",
        "sec_0037.jpg",
        "thi_0127.jpg",
        "thi_0128.jpg",
        "thi_0129.jpg"
    ]
    in_valid_list = [
        "0086(1).jpg",
        "0087(2).jpg",
        "0088(3).jpg",
        "hcjz_birads3_0056_0004(1).jpg",
        "hcjz_birads3_01170_0001(2).jpg",
        "hcjz_birads5_0002_0006(3).jpg",
        "sec_0035(1).jpg",
        "sec_0036(2).jpg",
        "sec_0037(3).jpg",
        "thi_0127(4).jpg",
        "thi_0128(5).jpg",
        "thi_0129(6).jpg",
        "0086.jpg__mixup__hcjz_birads3_0056_0004.jpg(1).jpg",
        "0087.jpg__mixup__thi_0128.jpg(1).jpg",
        "0087.jpg__mixup__mock_image.jpg(1).jpg",
        "mock_image.jpg__mixup__0087.jpg(1).jpg"
    ]
    in_valid_list = list(map(lambda x: os.path.join(os.curdir, 'data', 'breast', 'cla', 'augmented', x), in_valid_list))
    not_in_valid_list = [
        "0096(1).jpg",
        "0097(2).jpg",
        "0098(4).jpg",
        "hcjz_birads3_0056_0009(1).jpg",
        "hcjz_birads3_01170_0009(1).jpg",
        "hcjz_birads5_0002_0009(1).jpg",
        "thi_0147(1).jpg",
        "thi_0148(1).jpg",
        "thi_0149(1).jpg",
        "hcjz_birads3_0056_0009.jpg__mixup__thi_0149.jpg(1).jpg"
    ]
    not_in_valid_list = list(
        map(lambda x: os.path.join(os.curdir, 'data', 'breast', 'cla', 'augmented', x), not_in_valid_list))

    def test_in_valid(self):
        """
        测试能否正确识别出不能添加至训练集的图像
        :return:
        """
        for file_name in self.in_valid_list:
            self.assertTrue(in_valid(file_name, self.valid_list), file_name)

    def test_not_in_valid(self):
        """
        测试能否正确识别出能添加至训练集的图像
        :return:
        """
        for file_name in self.not_in_valid_list:
            self.assertFalse(in_valid(file_name, self.valid_list), file_name)

class TestLeakageMethod(unittest.TestCase):
    """
    用于测试测试信息泄露的方法是否正确
    """
    raw_train_image_without_augment_list = [
        ".\\data\\breast\\cla\\BUS\\Images\\bus_0101-l.png",
        ".\\data\\breast\\cla\\train\\hcjz_birads3_0051_0001.jpg",
        ".\\data\\breast\\cla\\train\\BUSI_0456.png",
        ".\\data\\breast\\cla\\train\\1056.jpg"
    ]
    train_image_without_augment_list = [
        "bus_0101-l",
        "hcjz_birads3_0051_0001",
        "BUSI_0456",
        "1056"
    ]
    raw_train_image_with_normal_augment_list = [
        ".\\data\\breast\\cla\\augmented\\Rotate,ratio=(1.0,1.0,1.0,1.0,1.0,1.0)-1\\0705(1).jpg",
        ".\\data\\breast\\cla\\augmented\\Rotate,ratio=(1.0,1.0,1.0,1.0,1.0,1.0)-1\\hcjz_birads3_0014_0003(1).jpg",
        ".\\data\\breast\\cla\\augmented\\Rotate,ratio=(1.0,1.0,1.0,1.0,1.0,1.0)-1\\bus_0814-s(1).png",
        ".\\data\\breast\\cla\\augmented\\Rotate,ratio=(1.0,1.0,1.0,1.0,1.0,1.0)-1\\case020(1).png"
    ]
    train_image_with_with_normal_augment_list = [
        "0705",
        "hcjz_birads3_0014_0003",
        "bus_0814-s",
        "case020"
    ]
    raw_valid_image_list = [
        ".\\data\\breast\\cla\\augmented\\Rotate,ratio=(1.0,1.0,1.0,1.0,1.0,1.0)-1\\0705.jpg",
        ".\\data\\breast\\cla\\augmented\\Rotate,ratio=(1.0,1.0,1.0,1.0,1.0,1.0)-1\\hcjz_birads3_0014_0003.jpg",
        ".\\data\\breast\\cla\\augmented\\Rotate,ratio=(1.0,1.0,1.0,1.0,1.0,1.0)-1\\bus_0814-s.png",
        ".\\data\\breast\\cla\\augmented\\Rotate,ratio=(1.0,1.0,1.0,1.0,1.0,1.0)-1\\case020.png"
    ]
    valid_image_list = [
        "0705",
        "hcjz_birads3_0014_0003",
        "bus_0814-s",
        "case020"
    ]
    raw_train_image_with_mixup_augment_list = [
        ".\\data\\breast\\fea\\augmented\\Mixup-1\\hcjz_malignant_0548_0011.jpg__mixup__hcjz_malignant_0650_0004.jpg.jpg",
        ".\\data\\breast\\fea\\train\\sec_0042.jpg",
        ".\\data\\breast\\fea\\augmented\\Mixup-1\\hcjz_birads4C_0075_0006.jpg__mixup__hcjz_benigh_0745_0003.jpg.jpg",
        ".\\data\\breast\\fea\\train\\thi_0212.jpg"
    ]
    train_image_with_with_mixup_augment_list = [
        ["hcjz_malignant_0548_0011","hcjz_malignant_0650_0004"],
        ["sec_0042"],
        ["hcjz_birads4C_0075_0006","hcjz_benigh_0745_0003"],
        ["thi_0212"]
    ]
    def test_test_image_without_augment(self):
        for raw,processed in zip(self.raw_train_image_without_augment_list,self.train_image_without_augment_list):
            self.assertEqual(processed,re.sub(r"\(\d\)","",raw.split(os.sep)[-1][:-4]))
    def test_test_image_with_normal_augment(self):
        for raw,processed in zip(self.raw_train_image_with_normal_augment_list,self.train_image_with_with_normal_augment_list):
            self.assertEqual(processed,re.sub(r"\(\d\)","",raw.split(os.sep)[-1][:-4]))
    def test_test_valid_image_without_augment(self):
        for raw, processed in zip(self.raw_valid_image_list, self.valid_image_list):
            self.assertEqual(processed, raw.split(os.sep)[-1][:-4])
    def test_test_image_with_mixup_augment(self):
        for raw, processed in zip(self.raw_train_image_with_mixup_augment_list,self.train_image_with_with_mixup_augment_list):
            self.assertEqual(processed, list(map(lambda x: x.replace('.png', '').replace('.jpg', ''),re.sub(r"\(\d\)", "", raw.split(os.sep)[-1][:-4]).split('__mixup__'))))


class ClaSplitAugmentedImageTestCase(unittest.TestCase):
    num_classes = 6
    valid_list = [
        "0086.jpg",
        "0087.jpg",
        "0088.jpg",
        "hcjz_birads3_0056_0004.jpg",
        "hcjz_birads3_01170_0001.jpg",
        "hcjz_birads5_0002_0006.jpg",
        "sec_0035.jpg",
        "sec_0036.jpg",
        "sec_0037.jpg",
        "thi_0127.jpg",
        "thi_0128.jpg",
        "thi_0129.jpg"
    ]
    valid_list = list(map(lambda x: os.path.join(os.curdir, 'data', 'breast', 'cla', 'train', x), valid_list))
    in_valid_list = [
        "0086(1).jpg",
        "0087(2).jpg",
        "0088(3).jpg",
        "hcjz_birads3_0056_0004(1).jpg",
        "hcjz_birads3_01170_0001(2).jpg",
        "hcjz_birads5_0002_0006(3).jpg",
        "sec_0035(1).jpg",
        "sec_0036(2).jpg",
        "sec_0037(3).jpg",
        "thi_0127(4).jpg",
        "thi_0128(5).jpg",
        "thi_0129(6).jpg",
        "0086.jpg__mixup__hcjz_birads3_0056_0004.jpg(1).jpg",
        "0087.jpg__mixup__thi_0128.jpg(1).jpg",
        "0087.jpg__mixup__mock_image.jpg(1).jpg",
        "mock_image.jpg__mixup__0087.jpg(1).jpg"
    ]
    not_in_valid_list = [
        "0096(1).jpg",
        "0097(2).jpg",
        "0098(4).jpg",
        "hcjz_birads3_0056_0009(1).jpg",
        "hcjz_birads3_01170_0009(1).jpg",
        "hcjz_birads5_0002_0009(1).jpg",
        "thi_0147(1).jpg",
        "thi_0148(1).jpg",
        "thi_0149(1).jpg",
        "hcjz_birads3_0056_0009.jpg__mixup__thi_0149.jpg(1).jpg"
    ]
    mock_valid_dataset = pd.DataFrame({'file_name': valid_list})
    mock_valid_dataset['label'] = 0
    mock_augmented_ground_truth = pd.DataFrame({'file_name': in_valid_list + not_in_valid_list})
    mock_augmented_ground_truth['label'] = 0
    mock_augmented_ground_truth = mock_augmented_ground_truth.sample(frac=1).reset_index(drop=True)

    @patch('pandas.read_csv')
    def test_in_valid(self, mock_read_csv):
        """
        测试由test_split_augmented_image返回的图像中，没有不可以加入训练集的
        :param mock_read_csv:
        :return:
        """
        mock_read_csv.return_value = self.mock_augmented_ground_truth.copy()
        not_in_valid_augmented_image = split_augmented_image(self.mock_valid_dataset, 'cla',[
            os.path.join(os.curdir, 'data', 'breast', 'cla', 'augmented')])
        self.assertEqual(0, len(np.intersect1d(not_in_valid_augmented_image.file_name, self.not_in_valid_list)))

    @patch('pandas.read_csv')
    def test_not_in_valid(self, mock_read_csv):
        """
        测试由test_split_augmented_image返回的图像中，包含了所有可以加入训练集的图像
        :param mock_read_csv:
        :return:
        """
        mock_read_csv.return_value = self.mock_augmented_ground_truth.copy()
        not_in_valid_augmented_image = split_augmented_image(self.mock_valid_dataset, 'cla',[
            os.path.join(os.curdir, 'data', 'breast', 'cla', 'augmented')])
        self.assertEqual(
            set([os.path.join(os.curdir, 'data', 'breast', 'cla', 'augmented', x) for x in self.not_in_valid_list]),
            set(not_in_valid_augmented_image.file_name.tolist()))
        self.assertEqual(len(self.not_in_valid_list), len(not_in_valid_augmented_image.values))


class GetBreastTrainValidDataTestCase(unittest.TestCase):
    valid_ratio = 0.3
    data_folder_path = os.path.join(os.curdir, 'data')
    if not len(glob(os.path.join(os.curdir, 'data','breast', 'cla','augmented','Mixup,ratio=(1.0,1.0,1.0,1.0,1.0,1.0)-*'))):
        MixUp(0.2,[1,1,1,1,1,1]).process_image()
    if not len(glob(os.path.join(os.curdir, 'data', 'breast', 'cla', 'augmented', 'Mixup,ratio=(2.0,1.0,3.0,4.0,5.0,6.0)-*'))):
        MixUp(0.2, [2,1,3,4,5,6]).process_image()
    if not len(glob(os.path.join(os.curdir, 'data', 'breast', 'cla', 'augmented','Rotate,ratio=(1.0,1.0,1.0,1.0,1.0,1.0)-*'))):
        Preprocess(A.Compose([A.Rotate(limit=15,p=1.0)]), [1, 1, 1, 1, 1, 1]).process_image()
    if not len(glob(os.path.join(os.curdir, 'data', 'breast', 'cla', 'augmented','Rotate,ratio=(2.0,1.0,3.0,4.0,5.0,6.0)-*'))):
        Preprocess(A.Compose([A.Rotate(limit=15,p=1.0)]), [2, 1, 3, 4, 5, 6]).process_image()
    if not len(glob(os.path.join(os.curdir, 'data', 'breast', 'fea', 'augmented','Mixup-*'))):
        MixUp(0.2, official_train=False,BUS=False,USG=False,fea_official_train=True).process_image()
    if not len(glob(os.path.join(os.curdir, 'data', 'breast', 'fea', 'augmented', 'VerticalFlip-*'))):
        Preprocess(A.Compose([A.VerticalFlip(p=1)]), official_train=False, BUS=False, USG=False, fea_official_train=True).process_image()

    cla_mixup_folder1 = glob(os.path.join(os.curdir, 'data','breast', 'cla','augmented','Mixup,ratio=(1.0,1.0,1.0,1.0,1.0,1.0)-*'))[0]
    cla_mixup_folder2 = glob(os.path.join(os.curdir, 'data','breast', 'cla','augmented','Mixup,ratio=(2.0,1.0,3.0,4.0,5.0,6.0)-*'))[0]
    cla_rotate_folder1 = glob(os.path.join(os.curdir, 'data','breast', 'cla','augmented','Rotate,ratio=(1.0,1.0,1.0,1.0,1.0,1.0)-*'))[0]
    cla_rotate_folder2 = glob(os.path.join(os.curdir, 'data','breast', 'cla','augmented','Rotate,ratio=(2.0,1.0,3.0,4.0,5.0,6.0)-*'))[0]
    fea_mixup_folder = glob(os.path.join(os.curdir, 'data', 'breast', 'fea', 'augmented','Mixup-*'))[0]
    fea_flip_folder = glob(os.path.join(os.curdir, 'data', 'breast', 'fea', 'augmented','VerticalFlip-*'))[0]


    def test_len(self):
        """
        测试数据集划分比例是否正确
        :return:
        """
        official_train, BUS, USG = True, True, True
        table = make_table(self.data_folder_path, official_train=official_train, BUS=BUS, USG=USG)
        train_dataset, valid_dataset = next(
            getBreastTrainValidData(self.data_folder_path, valid_ratio=self.valid_ratio, official_train=official_train,
                                    BUS=BUS, USG=USG))
        self.assertEqual(len(table), len(train_dataset) + len(valid_dataset))
        self.assertEqual(int(len(table) * self.valid_ratio), len(valid_dataset))

        official_train, BUS, USG = True, False, False
        table = make_table(self.data_folder_path, official_train=official_train, BUS=BUS, USG=USG)
        train_dataset, valid_dataset = next(
            getBreastTrainValidData(self.data_folder_path, valid_ratio=self.valid_ratio, official_train=official_train,
                                    BUS=BUS, USG=USG))
        self.assertEqual(len(table), len(train_dataset) + len(valid_dataset))
        self.assertEqual(int(len(table) * self.valid_ratio), len(valid_dataset))

        official_train, BUS, USG = False, True, False
        table = make_table(self.data_folder_path, official_train=official_train, BUS=BUS, USG=USG)
        train_dataset, valid_dataset = next(
            getBreastTrainValidData(self.data_folder_path, valid_ratio=self.valid_ratio, official_train=official_train,
                                    BUS=BUS, USG=USG))
        self.assertEqual(len(table), len(train_dataset) + len(valid_dataset))
        self.assertEqual(int(len(table) * self.valid_ratio), len(valid_dataset))

        official_train, BUS, USG = False, False, True
        table = make_table(self.data_folder_path, official_train=official_train, BUS=BUS, USG=USG)
        train_dataset, valid_dataset = next(
            getBreastTrainValidData(self.data_folder_path, valid_ratio=self.valid_ratio, official_train=official_train,
                                    BUS=BUS, USG=USG))
        self.assertEqual(len(table), len(train_dataset) + len(valid_dataset))
        self.assertEqual(int(len(table) * self.valid_ratio), len(valid_dataset))

        official_train, BUS, USG, fea_official_train = False, False, False, True
        table = make_table(self.data_folder_path, official_train=official_train, BUS=BUS, USG=USG,
                           fea_official_train=fea_official_train)
        train_dataset, valid_dataset = next(
            getBreastTrainValidData(self.data_folder_path, valid_ratio=self.valid_ratio, official_train=official_train,
                                    BUS=BUS, USG=USG, fea_official_train=fea_official_train))
        self.assertEqual(len(table), len(train_dataset) + len(valid_dataset))
        self.assertEqual(int(len(table) * self.valid_ratio), len(valid_dataset))

    def test_leakage_without_augment(self):
        """
        测试是否存在信息泄露
        :return:
        """
        official_train, BUS, USG = True, True, True
        train_dataset, valid_dataset = next(
            getBreastTrainValidData(self.data_folder_path, valid_ratio=self.valid_ratio, official_train=official_train,
                                    BUS=BUS, USG=USG))
        valid_list = valid_dataset.table.file_name.str.split(os.sep).str[-1].str[:-4].tolist()
        self.assertFalse(np.any(train_dataset.table.file_name.apply(lambda x:re.sub(r"\(\d\)","",x.split(os.sep)[-1][:-4])).apply(lambda x: any([valid == x for valid in valid_list]))))

        official_train, BUS, USG = True, False, False
        train_dataset, valid_dataset = next(
            getBreastTrainValidData(self.data_folder_path, valid_ratio=self.valid_ratio, official_train=official_train,
                                    BUS=BUS, USG=USG))
        valid_list = valid_dataset.table.file_name.str.split(os.sep).str[-1].str[:-4].tolist()
        self.assertFalse(np.any(train_dataset.table.file_name.apply(lambda x:re.sub(r"\(\d\)","",x.split(os.sep)[-1][:-4])).apply(lambda x: any([valid == x for valid in valid_list]))))

        official_train, BUS, USG = False, True, False
        train_dataset, valid_dataset = next(
            getBreastTrainValidData(self.data_folder_path, valid_ratio=self.valid_ratio, official_train=official_train,
                                    BUS=BUS, USG=USG))
        valid_list = valid_dataset.table.file_name.str.split(os.sep).str[-1].str[:-4].tolist()
        self.assertFalse(np.any(train_dataset.table.file_name.apply(lambda x:re.sub(r"\(\d\)","",x.split(os.sep)[-1][:-4])).apply(lambda x: any([valid == x for valid in valid_list]))))

        official_train, BUS, USG = False, False, True
        train_dataset, valid_dataset = next(
            getBreastTrainValidData(self.data_folder_path, valid_ratio=self.valid_ratio, official_train=official_train,
                                    BUS=BUS, USG=USG))
        valid_list = valid_dataset.table.file_name.str.split(os.sep).str[-1].str[:-4].tolist()
        self.assertFalse(np.any(train_dataset.table.file_name.apply(lambda x:re.sub(r"\(\d\)","",x.split(os.sep)[-1][:-4])).apply(lambda x: any([valid == x for valid in valid_list]))))

        official_train, BUS, USG, fea_official_train = False, False, False, True
        train_dataset, valid_dataset = next(
            getBreastTrainValidData(self.data_folder_path, valid_ratio=self.valid_ratio, official_train=official_train,
                                    BUS=BUS, USG=USG, fea_official_train=fea_official_train))
        valid_list = valid_dataset.table.file_name.str.split(os.sep).str[-1].str[:-4].tolist()
        self.assertFalse(np.any(train_dataset.table.file_name.apply(lambda x:re.sub(r"\(\d\)","",x.split(os.sep)[-1][:-4])).apply(lambda x: any([valid == x for valid in valid_list]))))

        official_train, BUS, USG = True, True, True
        train_dataset, valid_dataset = next(
            getBreastTrainValidData(self.data_folder_path, valid_ratio=self.valid_ratio, official_train=official_train,
                                    BUS=BUS, USG=USG,ratio=[1,1,1,1,1,1]))
        valid_list = valid_dataset.table.file_name.str.split(os.sep).str[-1].str[:-4].tolist()
        self.assertFalse(np.any(train_dataset.table.file_name.apply(lambda x:re.sub(r"\(\d\)","",x.split(os.sep)[-1][:-4])).apply(lambda x: any([valid == x for valid in valid_list]))))

        official_train, BUS, USG, fea_official_train = False, False, False, True
        train_dataset, valid_dataset = next(
            getBreastTrainValidData(self.data_folder_path, valid_ratio=self.valid_ratio, official_train=official_train,
                                    BUS=BUS, USG=USG, fea_official_train=fea_official_train,feature='boundary',ratio=[1,1]))
        valid_list = valid_dataset.table.file_name.str.split(os.sep).str[-1].str[:-4].tolist()
        self.assertFalse(np.any(train_dataset.table.file_name.apply(lambda x:re.sub(r"\(\d\)","",x.split(os.sep)[-1][:-4])).apply(lambda x: any([valid == x for valid in valid_list]))))

        official_train, BUS, USG, fea_official_train = False, False, False, True
        train_dataset, valid_dataset = next(
            getBreastTrainValidData(self.data_folder_path, valid_ratio=self.valid_ratio, official_train=official_train,
                                    BUS=BUS, USG=USG, fea_official_train=fea_official_train, feature='calcification',
                                    ratio=[1, 1]))
        valid_list = valid_dataset.table.file_name.str.split(os.sep).str[-1].str[:-4].tolist()
        self.assertFalse(np.any(train_dataset.table.file_name.apply(lambda x:re.sub(r"\(\d\)","",x.split(os.sep)[-1][:-4])).apply(lambda x: any([valid == x for valid in valid_list]))))

        official_train, BUS, USG, fea_official_train = False, False, False, True
        train_dataset, valid_dataset = next(
            getBreastTrainValidData(self.data_folder_path, valid_ratio=self.valid_ratio, official_train=official_train,
                                    BUS=BUS, USG=USG, fea_official_train=fea_official_train, feature='direction',
                                    ratio=[1, 1]))
        valid_list = valid_dataset.table.file_name.str.split(os.sep).str[-1].str[:-4].tolist()
        self.assertFalse(np.any(train_dataset.table.file_name.apply(lambda x:re.sub(r"\(\d\)","",x.split(os.sep)[-1][:-4])).apply(lambda x: any([valid == x for valid in valid_list]))))

        official_train, BUS, USG, fea_official_train = False, False, False, True
        train_dataset, valid_dataset = next(
            getBreastTrainValidData(self.data_folder_path, valid_ratio=self.valid_ratio, official_train=official_train,
                                    BUS=BUS, USG=USG, fea_official_train=fea_official_train, feature='shape',
                                    ratio=[1, 1]))
        valid_list = valid_dataset.table.file_name.str.split(os.sep).str[-1].str[:-4].tolist()
        self.assertFalse(np.any(train_dataset.table.file_name.apply(lambda x:re.sub(r"\(\d\)","",x.split(os.sep)[-1][:-4])).apply(lambda x: any([valid == x for valid in valid_list]))))

    def test_leakage_with_normal_augment(self):
        """
        测试是否存在信息泄露
        :return:
        """
        official_train, BUS, USG = True, True, True
        train_dataset, valid_dataset = next(
            getBreastTrainValidData(self.data_folder_path, valid_ratio=self.valid_ratio, official_train=official_train,
                                    BUS=BUS, USG=USG,augmented_folder_list=[self.cla_rotate_folder1]))
        valid_list = valid_dataset.table.file_name.str.split(os.sep).str[-1].str[:-4].tolist()
        self.assertFalse(np.any(
            train_dataset.table.file_name.apply(lambda x: re.sub(r"\(\d\)", "", x.split(os.sep)[-1][:-4])).apply(
                lambda x: any([valid == x for valid in valid_list]))))

        official_train, BUS, USG = True, True, True
        train_dataset, valid_dataset = next(
            getBreastTrainValidData(self.data_folder_path, valid_ratio=self.valid_ratio, official_train=official_train,
                                    BUS=BUS, USG=USG,augmented_folder_list=[self.cla_rotate_folder1,self.cla_rotate_folder2]))
        valid_list = valid_dataset.table.file_name.str.split(os.sep).str[-1].str[:-4].tolist()
        self.assertFalse(np.any(
            train_dataset.table.file_name.apply(lambda x: re.sub(r"\(\d\)", "", x.split(os.sep)[-1][:-4])).apply(
                lambda x: any([valid == x for valid in valid_list]))))

        official_train, BUS, USG = True, False, False
        train_dataset, valid_dataset = next(
            getBreastTrainValidData(self.data_folder_path, valid_ratio=self.valid_ratio, official_train=official_train,
                                    BUS=BUS, USG=USG,augmented_folder_list=[self.cla_rotate_folder1]))
        valid_list = valid_dataset.table.file_name.str.split(os.sep).str[-1].str[:-4].tolist()
        self.assertFalse(np.any(
            train_dataset.table.file_name.apply(lambda x: re.sub(r"\(\d\)", "", x.split(os.sep)[-1][:-4])).apply(
                lambda x: any([valid == x for valid in valid_list]))))

        official_train, BUS, USG = False, True, False
        train_dataset, valid_dataset = next(
            getBreastTrainValidData(self.data_folder_path, valid_ratio=self.valid_ratio, official_train=official_train,
                                    BUS=BUS, USG=USG,augmented_folder_list=[self.cla_rotate_folder1]))
        valid_list = valid_dataset.table.file_name.str.split(os.sep).str[-1].str[:-4].tolist()
        self.assertFalse(np.any(
            train_dataset.table.file_name.apply(lambda x: re.sub(r"\(\d\)", "", x.split(os.sep)[-1][:-4])).apply(
                lambda x: any([valid == x for valid in valid_list]))))

        official_train, BUS, USG = False, False, True
        train_dataset, valid_dataset = next(
            getBreastTrainValidData(self.data_folder_path, valid_ratio=self.valid_ratio, official_train=official_train,
                                    BUS=BUS, USG=USG,augmented_folder_list=[self.cla_rotate_folder1]))
        valid_list = valid_dataset.table.file_name.str.split(os.sep).str[-1].str[:-4].tolist()
        self.assertFalse(np.any(
            train_dataset.table.file_name.apply(lambda x: re.sub(r"\(\d\)", "", x.split(os.sep)[-1][:-4])).apply(
                lambda x: any([valid == x for valid in valid_list]))))

        official_train, BUS, USG, fea_official_train = False, False, False, True
        train_dataset, valid_dataset = next(
            getBreastTrainValidData(self.data_folder_path, valid_ratio=self.valid_ratio, official_train=official_train,
                                    BUS=BUS, USG=USG, fea_official_train=fea_official_train,augmented_folder_list=[self.fea_flip_folder]))
        valid_list = valid_dataset.table.file_name.str.split(os.sep).str[-1].str[:-4].tolist()
        self.assertFalse(np.any(
            train_dataset.table.file_name.apply(lambda x: re.sub(r"\(\d\)", "", x.split(os.sep)[-1][:-4])).apply(
                lambda x: any([valid == x for valid in valid_list]))))



        official_train, BUS, USG = True, True, True
        train_dataset, valid_dataset = next(
            getBreastTrainValidData(self.data_folder_path, valid_ratio=self.valid_ratio, official_train=official_train,
                                    BUS=BUS, USG=USG,ratio=[1,1,1,1,1,1],augmented_folder_list=[self.cla_rotate_folder1,self.cla_rotate_folder2]))
        valid_list = valid_dataset.table.file_name.str.split(os.sep).str[-1].str[:-4].tolist()
        self.assertFalse(np.any(
            train_dataset.table.file_name.apply(lambda x: re.sub(r"\(\d\)", "", x.split(os.sep)[-1][:-4])).apply(
                lambda x: any([valid == x for valid in valid_list]))))

        official_train, BUS, USG, fea_official_train = False, False, False, True
        train_dataset, valid_dataset = next(
            getBreastTrainValidData(self.data_folder_path, valid_ratio=self.valid_ratio, official_train=official_train,
                                    BUS=BUS, USG=USG, fea_official_train=fea_official_train,feature='boundary',ratio=[1,1],augmented_folder_list=[self.fea_flip_folder]))
        valid_list = valid_dataset.table.file_name.str.split(os.sep).str[-1].str[:-4].tolist()
        self.assertFalse(np.any(
            train_dataset.table.file_name.apply(lambda x: re.sub(r"\(\d\)", "", x.split(os.sep)[-1][:-4])).apply(
                lambda x: any([valid == x for valid in valid_list]))))



    def test_leakage_with_mixup_augment(self):
        official_train, BUS, USG, fea_official_train = False, False, False, True
        train_dataset, valid_dataset = next(
            getBreastTrainValidData(self.data_folder_path, valid_ratio=self.valid_ratio, official_train=official_train,
                                    BUS=BUS, USG=USG, fea_official_train=fea_official_train, feature='calcification',
                                    ratio=[1, 1],augmented_folder_list=[self.fea_mixup_folder]))
        valid_list = valid_dataset.table.file_name.str.split(os.sep).str[-1].str[:-4].tolist()
        self.assertFalse(np.any(train_dataset.table.file_name.apply(lambda x: list(
            map(lambda x: x.replace('.png', '').replace('.jpg', ''),
                re.sub(r"\(\d\)", "", x.split(os.sep)[-1][:-4]).split('__mixup__')))).apply(
            lambda x: any([valid in x for valid in valid_list]))))

        official_train, BUS, USG, fea_official_train = False, False, False, True
        train_dataset, valid_dataset = next(
            getBreastTrainValidData(self.data_folder_path, valid_ratio=self.valid_ratio, official_train=official_train,
                                    BUS=BUS, USG=USG, fea_official_train=fea_official_train, feature='direction',
                                    ratio=[1, 1],augmented_folder_list=[self.fea_flip_folder,self.fea_mixup_folder]))
        valid_list = valid_dataset.table.file_name.str.split(os.sep).str[-1].str[:-4].tolist()
        self.assertFalse(np.any(train_dataset.table.file_name.apply(lambda x: list(
            map(lambda x: x.replace('.png', '').replace('.jpg', ''),
                re.sub(r"\(\d\)", "", x.split(os.sep)[-1][:-4]).split('__mixup__')))).apply(
            lambda x: any([valid in x for valid in valid_list]))))

        official_train, BUS, USG, fea_official_train = False, False, False, True
        train_dataset, valid_dataset = next(
            getBreastTrainValidData(self.data_folder_path, valid_ratio=self.valid_ratio, official_train=official_train,
                                    BUS=BUS, USG=USG, fea_official_train=fea_official_train, feature='shape',
                                    ratio=[1, 1],augmented_folder_list=[self.fea_flip_folder,self.fea_mixup_folder]))
        valid_list = valid_dataset.table.file_name.str.split(os.sep).str[-1].str[:-4].tolist()
        self.assertFalse(np.any(train_dataset.table.file_name.apply(lambda x: list(
            map(lambda x: x.replace('.png', '').replace('.jpg', ''),
                re.sub(r"\(\d\)", "", x.split(os.sep)[-1][:-4]).split('__mixup__')))).apply(
            lambda x: any([valid in x for valid in valid_list]))))


        official_train, BUS, USG, fea_official_train = False, False, False, True
        train_dataset, valid_dataset = next(
            getBreastTrainValidData(self.data_folder_path, valid_ratio=self.valid_ratio, official_train=official_train,
                                    BUS=BUS, USG=USG, fea_official_train=fea_official_train,augmented_folder_list=[self.fea_mixup_folder]))
        valid_list = valid_dataset.table.file_name.str.split(os.sep).str[-1].str[:-4].tolist()
        self.assertFalse(np.any(train_dataset.table.file_name.apply(lambda x: list(
            map(lambda x: x.replace('.png', '').replace('.jpg', ''),
                re.sub(r"\(\d\)", "", x.split(os.sep)[-1][:-4]).split('__mixup__')))).apply(
            lambda x: any([valid in x for valid in valid_list]))))

    def test_valid_source(self):
        """
        测试验证集中没有通过数据增广得到的数据
        :return:
        """
        official_train, BUS, USG = True, True, True
        train_dataset, valid_dataset = next(
            getBreastTrainValidData(self.data_folder_path, valid_ratio=self.valid_ratio, official_train=official_train,
                                    BUS=BUS, USG=USG,augmented_folder_list=[self.cla_rotate_folder1]))
        valid_list = valid_dataset.table.file_name.str.split(os.sep).str[-1].str[:-4].tolist()
        self.assertFalse(any(['('in x for x in valid_list]))

        official_train, BUS, USG = True, False, False
        train_dataset, valid_dataset = next(
            getBreastTrainValidData(self.data_folder_path, valid_ratio=self.valid_ratio, official_train=official_train,
                                    BUS=BUS, USG=USG))
        valid_list = valid_dataset.table.file_name.str.split(os.sep).str[-1].str[:-4].tolist()
        self.assertFalse(any(['('in x for x in valid_list]))

        official_train, BUS, USG = False, True, False
        train_dataset, valid_dataset = next(
            getBreastTrainValidData(self.data_folder_path, valid_ratio=self.valid_ratio, official_train=official_train,
                                    BUS=BUS, USG=USG))
        valid_list = valid_dataset.table.file_name.str.split(os.sep).str[-1].str[:-4].tolist()
        self.assertFalse(any(['('in x for x in valid_list]))

        official_train, BUS, USG = False, False, True
        train_dataset, valid_dataset = next(
            getBreastTrainValidData(self.data_folder_path, valid_ratio=self.valid_ratio, official_train=official_train,
                                    BUS=BUS, USG=USG))
        valid_list = valid_dataset.table.file_name.str.split(os.sep).str[-1].str[:-4].tolist()
        self.assertFalse(any(['('in x for x in valid_list]))

        official_train, BUS, USG, fea_official_train = False, False, False, True
        train_dataset, valid_dataset = next(
            getBreastTrainValidData(self.data_folder_path, valid_ratio=self.valid_ratio, official_train=official_train,
                                    BUS=BUS, USG=USG, fea_official_train=fea_official_train))
        valid_list = valid_dataset.table.file_name.str.split(os.sep).str[-1].str[:-4].tolist()
        self.assertFalse(any(['('in x for x in valid_list]))

        official_train, BUS, USG = True, True, True
        train_dataset, valid_dataset = next(
            getBreastTrainValidData(self.data_folder_path, valid_ratio=self.valid_ratio, official_train=official_train,
                                    BUS=BUS, USG=USG,ratio=[1,1,1,1,1,1]))
        valid_list = valid_dataset.table.file_name.str.split(os.sep).str[-1].str[:-4].tolist()
        self.assertFalse(any(['('in x for x in valid_list]))

        official_train, BUS, USG, fea_official_train = False, False, False, True
        train_dataset, valid_dataset = next(
            getBreastTrainValidData(self.data_folder_path, valid_ratio=self.valid_ratio, official_train=official_train,
                                    BUS=BUS, USG=USG, fea_official_train=fea_official_train,feature='boundary',ratio=[1,1]))
        valid_list = valid_dataset.table.file_name.str.split(os.sep).str[-1].str[:-4].tolist()
        self.assertFalse(any(['('in x for x in valid_list]))

        official_train, BUS, USG, fea_official_train = False, False, False, True
        train_dataset, valid_dataset = next(
            getBreastTrainValidData(self.data_folder_path, valid_ratio=self.valid_ratio, official_train=official_train,
                                    BUS=BUS, USG=USG, fea_official_train=fea_official_train, feature='calcification',
                                    ratio=[1, 1]))
        valid_list = valid_dataset.table.file_name.str.split(os.sep).str[-1].str[:-4].tolist()
        self.assertFalse(any(['('in x for x in valid_list]))

        official_train, BUS, USG, fea_official_train = False, False, False, True
        train_dataset, valid_dataset = next(
            getBreastTrainValidData(self.data_folder_path, valid_ratio=self.valid_ratio, official_train=official_train,
                                    BUS=BUS, USG=USG, fea_official_train=fea_official_train, feature='direction',
                                    ratio=[1, 1]))
        valid_list = valid_dataset.table.file_name.str.split(os.sep).str[-1].str[:-4].tolist()
        self.assertFalse(any(['('in x for x in valid_list]))

        official_train, BUS, USG, fea_official_train = False, False, False, True
        train_dataset, valid_dataset = next(
            getBreastTrainValidData(self.data_folder_path, valid_ratio=self.valid_ratio, official_train=official_train,
                                    BUS=BUS, USG=USG, fea_official_train=fea_official_train, feature='shape',
                                    ratio=[1, 1]))
        valid_list = valid_dataset.table.file_name.str.split(os.sep).str[-1].str[:-4].tolist()
        self.assertFalse(any(['('in x for x in valid_list]))

        official_train, BUS, USG = True, True, True
        train_dataset, valid_dataset = next(
            getBreastTrainValidData(self.data_folder_path, valid_ratio=self.valid_ratio, official_train=official_train,
                                    BUS=BUS, USG=USG,augmented_folder_list=[self.cla_rotate_folder1]))
        valid_list = valid_dataset.table.file_name.str.split(os.sep).str[-1].str[:-4].tolist()
        self.assertFalse(any(['('in x for x in valid_list]))


        official_train, BUS, USG = True, True, True
        train_dataset, valid_dataset = next(
            getBreastTrainValidData(self.data_folder_path, valid_ratio=self.valid_ratio, official_train=official_train,
                                    BUS=BUS, USG=USG,augmented_folder_list=[self.cla_rotate_folder1,self.cla_rotate_folder2]))
        valid_list = valid_dataset.table.file_name.str.split(os.sep).str[-1].str[:-4].tolist()
        self.assertFalse(any(['('in x for x in valid_list]))

        official_train, BUS, USG = True, False, False
        train_dataset, valid_dataset = next(
            getBreastTrainValidData(self.data_folder_path, valid_ratio=self.valid_ratio, official_train=official_train,
                                    BUS=BUS, USG=USG,augmented_folder_list=[self.cla_rotate_folder1]))
        valid_list = valid_dataset.table.file_name.str.split(os.sep).str[-1].str[:-4].tolist()
        self.assertFalse(any(['('in x for x in valid_list]))

        official_train, BUS, USG = False, True, False
        train_dataset, valid_dataset = next(
            getBreastTrainValidData(self.data_folder_path, valid_ratio=self.valid_ratio, official_train=official_train,
                                    BUS=BUS, USG=USG,augmented_folder_list=[self.cla_rotate_folder1]))
        valid_list = valid_dataset.table.file_name.str.split(os.sep).str[-1].str[:-4].tolist()
        self.assertFalse(any(['('in x for x in valid_list]))

        official_train, BUS, USG = False, False, True
        train_dataset, valid_dataset = next(
            getBreastTrainValidData(self.data_folder_path, valid_ratio=self.valid_ratio, official_train=official_train,
                                    BUS=BUS, USG=USG,augmented_folder_list=[self.cla_rotate_folder1]))
        valid_list = valid_dataset.table.file_name.str.split(os.sep).str[-1].str[:-4].tolist()
        self.assertFalse(any(['('in x for x in valid_list]))

        official_train, BUS, USG, fea_official_train = False, False, False, True
        train_dataset, valid_dataset = next(
            getBreastTrainValidData(self.data_folder_path, valid_ratio=self.valid_ratio, official_train=official_train,
                                    BUS=BUS, USG=USG, fea_official_train=fea_official_train,augmented_folder_list=[self.fea_flip_folder]))
        valid_list = valid_dataset.table.file_name.str.split(os.sep).str[-1].str[:-4].tolist()
        self.assertFalse(any(['('in x for x in valid_list]))

        official_train, BUS, USG, fea_official_train = False, False, False, True
        train_dataset, valid_dataset = next(
            getBreastTrainValidData(self.data_folder_path, valid_ratio=self.valid_ratio, official_train=official_train,
                                    BUS=BUS, USG=USG, fea_official_train=fea_official_train, feature='calcification',
                                    ratio=[1, 1], augmented_folder_list=[self.fea_mixup_folder]))
        valid_list = valid_dataset.table.file_name.str.split(os.sep).str[-1].str[:-4].tolist()
        self.assertFalse(any(['('in x for x in valid_list]))

        official_train, BUS, USG, fea_official_train = False, False, False, True
        train_dataset, valid_dataset = next(
            getBreastTrainValidData(self.data_folder_path, valid_ratio=self.valid_ratio, official_train=official_train,
                                    BUS=BUS, USG=USG, fea_official_train=fea_official_train, feature='direction',
                                    ratio=[1, 1], augmented_folder_list=[self.fea_flip_folder, self.fea_mixup_folder]))
        valid_list = valid_dataset.table.file_name.str.split(os.sep).str[-1].str[:-4].tolist()
        self.assertFalse(any(['('in x for x in valid_list]))

        official_train, BUS, USG, fea_official_train = False, False, False, True
        train_dataset, valid_dataset = next(
            getBreastTrainValidData(self.data_folder_path, valid_ratio=self.valid_ratio, official_train=official_train,
                                    BUS=BUS, USG=USG, fea_official_train=fea_official_train, feature='shape',
                                    ratio=[1, 1], augmented_folder_list=[self.fea_flip_folder, self.fea_mixup_folder]))
        valid_list = valid_dataset.table.file_name.str.split(os.sep).str[-1].str[:-4].tolist()
        self.assertFalse(any(['('in x for x in valid_list]))

        official_train, BUS, USG, fea_official_train = False, False, False, True
        train_dataset, valid_dataset = next(
            getBreastTrainValidData(self.data_folder_path, valid_ratio=self.valid_ratio, official_train=official_train,
                                    BUS=BUS, USG=USG, fea_official_train=fea_official_train,
                                    augmented_folder_list=[self.fea_mixup_folder]))
        valid_list = valid_dataset.table.file_name.str.split(os.sep).str[-1].str[:-4].tolist()
        self.assertFalse(any(['('in x for x in valid_list]))

class ClaMakeTableTestCase(unittest.TestCase):
    num_official_train = 2210 + 217  # jpg+png
    num_BUS = 0 + 1875 - 693  # jpg+png-4类
    num_USG = 0 + 256 - 4  # jpg+png-1类
    data_folder_path = os.path.join(os.curdir, 'data')
    num_classes = 6
    num_BUS_classes = 3  # 舍去4a，4b，4c

    def test_columns(self):
        """
        测试table是否有file_name和label列
        :return:
        """
        table = make_table(self.data_folder_path, official_train=True, BUS=True, USG=True)
        columns = table.columns
        self.assertIn('file_name', columns)
        self.assertIn('label', columns)

    def test_labels(self):
        """
        测试label的个数和种类数量
        :return:
        """
        table = make_table(self.data_folder_path, official_train=True, BUS=True, USG=True)
        self.assertTrue(np.all(table.label.isin(np.arange(self.num_classes))))

        table = make_table(self.data_folder_path, official_train=True, BUS=False, USG=False)
        self.assertEqual(self.num_classes, table.label.nunique(), table.label.unique())

        table = make_table(self.data_folder_path, official_train=False, BUS=True, USG=False)
        self.assertEqual(self.num_BUS_classes, table.label.nunique(), table.label.unique())

        table = make_table(self.data_folder_path, official_train=False, BUS=False, USG=True)
        self.assertEqual(self.num_classes, table.label.nunique(), table.label.unique())

    def test_file(self):
        """
        测试是否所有file_name路径都存在
        :return:
        """
        table = make_table(self.data_folder_path, official_train=True, BUS=True, USG=True)
        self.assertTrue(np.all(table.file_name.apply(os.path.exists)))

    def test_num_sample(self):
        """
        测试样本数量是否正确
        :return:
        """
        table = make_table(self.data_folder_path, official_train=True, BUS=False, USG=False)
        self.assertEqual(self.num_official_train, len(table))

        table = make_table(self.data_folder_path, official_train=False, BUS=True, USG=False)
        self.assertEqual(self.num_BUS, len(table))

        table = make_table(self.data_folder_path, official_train=False, BUS=False, USG=True)
        self.assertEqual(self.num_USG, len(table))

        table = make_table(self.data_folder_path, official_train=True, BUS=True, USG=True)
        self.assertEqual(self.num_USG + self.num_BUS + self.num_official_train, len(table))


class ClaTableDatasetTestCase(unittest.TestCase):
    test_images = os.listdir(os.path.join(os.curdir, 'data', 'test'))
    num_classes = 6
    seed = 42
    labels = np.random.randint(low=0, high=num_classes, size=len(test_images))
    pesu_table = pd.DataFrame({
        "file_name": [os.path.join(os.curdir, 'data', 'test', test_image) for test_image in test_images],
        "label": labels
    })
    transform = Compose([RandomCrop((128, 128)), RandomHorizontalFlip()])

    def test_PIL_format_without_transform(self):
        """
        测试读取返回的图像的格式
        :return:
        """
        tableDataset = TableDataset(table=self.pesu_table, image_format='PIL')
        for image, label in tableDataset:
            self.assertIsInstance(image, Image)

    def test_Tensor_format_without_transform(self):
        """
        测试读取返回的图像的格式
        :return:
        """
        tableDataset = TableDataset(table=self.pesu_table, image_format='Tensor')
        for image, label in tableDataset:
            self.assertIsInstance(image, Tensor)

    def test_ndarray_format_without_transform(self):
        """
        测试读取返回的图像的格式
        :return:
        """
        tableDataset = TableDataset(table=self.pesu_table, image_format='ndarray')
        for image, label in tableDataset:
            self.assertIsInstance(image, ndarray)

    def test_label_without_transform(self):
        """
        测试label是否正确
        :return:
        """
        tableDataset = TableDataset(table=self.pesu_table, image_format='ndarray')
        idx = 0
        for image, label in tableDataset:
            self.assertEqual(label, self.labels[idx])
            idx += 1

    def test_PIL_format_with_transform(self):
        """
        测试读取返回的图像的格式
        :return:
        """
        tableDataset = TableDataset(table=self.pesu_table, image_format='PIL', transform=self.transform)
        for image, label in tableDataset:
            self.assertIsInstance(image, Image)

    def test_Tensor_format_with_transform(self):
        """
        测试读取返回的图像的格式
        :return:
        """
        tableDataset = TableDataset(table=self.pesu_table, image_format='Tensor', transform=self.transform)
        for image, label in tableDataset:
            self.assertIsInstance(image, Tensor)

    def test_label_with_transform(self):
        """
        测试label是否正确
        :return:
        """
        tableDataset = TableDataset(table=self.pesu_table, image_format='PIL', transform=self.transform)
        idx = 0
        for image, label in tableDataset:
            self.assertEqual(self.labels[idx], label)
            idx += 1


class MyBCELossTestCase(unittest.TestCase):
    myBCELoss = MyBCELoss()
    sigmoid = torch.nn.Sigmoid()

    def test_MyBCELoss1(self):
        """
        测试BCELoss
        :return:
        """
        input = torch.Tensor([
            [1],
        ])
        target = torch.Tensor([
            [0],
        ])
        result = - (target * torch.log(self.sigmoid(input)) + (1 - target) * torch.log(
            1 - self.sigmoid(input))).mean().item()
        self.assertEqual(result, self.myBCELoss(input=input, target=target).item())

    def test_MyBCELoss2(self):
        """
        测试BCELoss
        :return:
        """
        input = torch.Tensor([
            [0],
        ])
        target = torch.Tensor([
            [0],
        ])
        result = - (target * torch.log(self.sigmoid(input)) + (1 - target) * torch.log(
            1 - self.sigmoid(input))).mean().item()
        self.assertEqual(result, self.myBCELoss(input=input, target=target).item())

    def test_MyBCELoss3(self):
        """
        测试MyBCE
        :return:
        """
        input = torch.Tensor([
            [1, 2, 3, 4],
            [-1, -2, -3, -4]
        ])
        target = torch.Tensor([
            [0, 0, 1, 1],
            [0, 0, 0, 0]
        ])
        result = - (target * torch.log(self.sigmoid(input)) + (1 - target) * torch.log(
            1 - self.sigmoid(input))).mean().item()
        self.assertEqual(result, self.myBCELoss(input=input, target=target).item())


class MyMultiLabelAccuracyTestCase(unittest.TestCase):
    def test_MyMultiLabelAccuracy(self):
        """
        fea任务Accuracy计算
        :return:
        """
        input = torch.Tensor([
            [1, -1, 1, 1],
            [-1, -1, 1, -1],
            [1, 1, 1, -1],
            [1, 1, -1, 1]
        ])
        target = torch.Tensor([
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 1]
        ])
        self.assertAlmostEqual(0.75, my_multilabel_accuracy(input, target))


class MyMultiLabelF1ScoreTestCase(unittest.TestCase):
    def test_MyMultiLabelF1Score(self):
        """
        fea任务Accuracy计算
        :return:
        """
        input = torch.Tensor([
            [1, -1, 1, 1],
            [-1, -1, 1, -1],
            [1, 1, 1, -1],
            [1, 1, -1, 1]
        ])
        target = torch.Tensor([
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 1]
        ])
        self.assertAlmostEqual(0.775, multilabel_f1_score(input, target))


class MyMultiLabelConfusionMatrixTestCase(unittest.TestCase):
    def test_MyMultiLabelConfusionMatrix(self):
        """
        fea任务Accuracy计算
        :return:
        """
        input = torch.Tensor([
            [1, -1, 1, 1],
            [-1, -1, 1, -1],
            [1, 1, 1, -1],
            [1, 1, -1, 1]
        ])
        target = torch.Tensor([
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 1]
        ])
        self.assertTrue(torch.all(torch.Tensor([
            [
                [3, 0],
                [0, 1]
            ],
            [
                [1, 1],
                [1, 1]
            ],
            [
                [2, 0],
                [1, 1]
            ],
            [
                [2, 1],
                [0, 1]
            ],
        ]) == multilabel_confusion_matrix(input, target)).item(),f"got {multilabel_confusion_matrix(input, target)}")

if __name__ == '__main__':
    unittest.main()
