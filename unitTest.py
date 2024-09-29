import random
import unittest
import pandas as pd
import numpy as np
import os
from PIL.Image import Image
from torch import Tensor
from numpy import ndarray
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip
from utils.ClaDataset import make_table, getClaTrainValidData
from utils.TableDataset import TableDataset


class SplitAugmentedImageTestCase(unittest.TestCase):
    num_classes = 6
    valid_file_name = [
        os.path.join(os.pardir,os.pardir,"mock_path","a.jpg"),
        os.path.join(os.pardir,os.pardir,"mock_path","b.jpg"),
        os.path.join(os.pardir,os.pardir,"mock_path","c.jpg"),
        os.path.join(os.pardir,os.pardir,"mock_path","d.jpg"),
        os.path.join(os.pardir,os.pardir,"mock_path","e.jpg"),
    ]
    augmented_file_name = [
        "a(1).jpg",
        "c(5).jpg",
        "e(3).jpg"
    ]
    labels = np.random.randint(low=0, high=num_classes, size=len(valid_file_name))
    mock_valid_set = pd.DataFrame({
        "file_name": valid_file_name,
        "label": labels
    })
    mock_augmented_set = pd.DataFrame({
        "file_name": augmented_file_name,
        "label": labels
    })
    def test_no_mixup_case:



class GetClaTrainValidDataTestCase(unittest.TestCase):
    valid_ratio = 0.3
    data_folder_path = os.path.join(os.curdir, 'data')

    def test_len(self):
        """
        测试数据集划分比例是否正确
        :return:
        """
        official_train, BUS, USG = True, True, True
        table = make_table(self.data_folder_path, official_train=official_train, BUS=BUS, USG=USG)
        train_dataset, valid_dataset = next(
            getClaTrainValidData(self.data_folder_path, valid_ratio=self.valid_ratio, official_train=official_train,
                                 BUS=BUS, USG=USG))
        self.assertEqual(len(table), len(train_dataset) + len(valid_dataset))
        self.assertEqual(int(len(table) * self.valid_ratio), len(valid_dataset))

        official_train, BUS, USG = True, False, False
        table = make_table(self.data_folder_path, official_train=official_train, BUS=BUS, USG=USG)
        train_dataset, valid_dataset = next(
            getClaTrainValidData(self.data_folder_path, valid_ratio=self.valid_ratio, official_train=official_train,
                                 BUS=BUS, USG=USG))
        self.assertEqual(len(table), len(train_dataset) + len(valid_dataset))
        self.assertEqual(int(len(table) * self.valid_ratio), len(valid_dataset))

        official_train, BUS, USG = False, True, False
        table = make_table(self.data_folder_path, official_train=official_train, BUS=BUS, USG=USG)
        train_dataset, valid_dataset = next(
            getClaTrainValidData(self.data_folder_path, valid_ratio=self.valid_ratio, official_train=official_train,
                                 BUS=BUS, USG=USG))
        self.assertEqual(len(table), len(train_dataset) + len(valid_dataset))
        self.assertEqual(int(len(table) * self.valid_ratio), len(valid_dataset))

        official_train, BUS, USG = False, False, True
        table = make_table(self.data_folder_path, official_train=official_train, BUS=BUS, USG=USG)
        train_dataset, valid_dataset = next(
            getClaTrainValidData(self.data_folder_path, valid_ratio=self.valid_ratio, official_train=official_train,
                                 BUS=BUS, USG=USG))
        self.assertEqual(len(table), len(train_dataset) + len(valid_dataset))
        self.assertEqual(int(len(table) * self.valid_ratio), len(valid_dataset))

    def test_leakage(self):
        """
        测试是否存在信息泄露
        :return:
        """
        official_train, BUS, USG = True, True, True
        table = make_table(self.data_folder_path, official_train=official_train, BUS=BUS, USG=USG)
        train_dataset, valid_dataset = next(
            getClaTrainValidData(self.data_folder_path, valid_ratio=self.valid_ratio, official_train=official_train,
                                 BUS=BUS, USG=USG))
        self.assertEqual(0, len(np.intersect1d(train_dataset.table.file_name, valid_dataset.table.file_name)))

        official_train, BUS, USG = True, False, False
        table = make_table(self.data_folder_path, official_train=official_train, BUS=BUS, USG=USG)
        train_dataset, valid_dataset = next(
            getClaTrainValidData(self.data_folder_path, valid_ratio=self.valid_ratio, official_train=official_train,
                                 BUS=BUS, USG=USG))
        self.assertEqual(0, len(np.intersect1d(train_dataset.table.file_name, valid_dataset.table.file_name)))

        official_train, BUS, USG = False, True, False
        table = make_table(self.data_folder_path, official_train=official_train, BUS=BUS, USG=USG)
        train_dataset, valid_dataset = next(
            getClaTrainValidData(self.data_folder_path, valid_ratio=self.valid_ratio, official_train=official_train,
                                 BUS=BUS, USG=USG))
        self.assertEqual(0, len(np.intersect1d(train_dataset.table.file_name, valid_dataset.table.file_name)))

        official_train, BUS, USG = False, False, True
        table = make_table(self.data_folder_path, official_train=official_train, BUS=BUS, USG=USG)
        train_dataset, valid_dataset = next(
            getClaTrainValidData(self.data_folder_path, valid_ratio=self.valid_ratio, official_train=official_train,
                                 BUS=BUS, USG=USG))
        self.assertEqual(0, len(np.intersect1d(train_dataset.table.file_name, valid_dataset.table.file_name)))


class MakeTableTestCase(unittest.TestCase):
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


class TableDatasetTestCase(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
