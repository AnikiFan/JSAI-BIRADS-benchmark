import pandas as pd
from logging import warning
import os
from tqdm import tqdm
import shutil


class OfficialFeaDataOrganizer:
    """
    针对官方Fea数据集进行整理
    同时带有筛出不符合规范的label.txt的功能
    用于汇总图像至一个文件夹中，并形成ground_truth.csv
    """
    feature_folder = ["boundary_labels", "calcification_labels", "direction_labels", "shape_labels"]

    def __init__(self, src: str):
        """
        :param src: boundary_labels……文件夹所在目录
        :param dst: 整理后存储的目录
        """
        self.src = src
        if os.path.exists(os.path.join(self.src, "images","ground_truth.csv")):
            os.remove(os.path.join(self.src, "images","ground_truth.csv"))
        if os.path.exists(os.path.join(self.src, "images", "fea_order.csv")):
            os.remove(os.path.join(self.src, "images", "fea_order.csv"))
        self.images = os.listdir(os.path.join(self.src, "images"))

    def check(self, file_name):
        if not all([os.path.exists(os.path.join(self.src, folder, file_name[:-4]+".txt")) for folder in
                    OfficialFeaDataOrganizer.feature_folder]):
            warning(f"label of {file_name} missing")
            return False
        for label_file in [os.path.join(self.src, folder, file_name[:-4]+".txt") for folder in
                           OfficialFeaDataOrganizer.feature_folder]:
            with open(label_file, "r") as f:
                content = list(filter(lambda x:x.strip(),f.readlines()))
                if len(content) > 1:
                    warning(f"label file {label_file} has multiple lines")
                    return False
                if len(content) == 0:
                    warning(f"label file {label_file} is empty")
                    return False
            if len(content[0].split(' ')) != 5:
                warning(f"label file {label_file} has wrong format")
                return False
            if not content[0].split(' ')[0] in ['0', '1']:
                warning(f"label file {label_file} has wrong label")
                return False
        return True

    def collect_labels(self, file_name):
        labels = []
        for label_file in [os.path.join(self.src, folder, file_name[:-4]+".txt") for folder in
                           OfficialFeaDataOrganizer.feature_folder]:
            with open(label_file, "r") as f:
                content = f.readline()[0]
                labels.append(content.split(' ')[0])
        return ''.join(labels)

    def organize(self, ignore):
        if not ignore:
            self.images = filter(self.check, self.images)
        table = pd.DataFrame({"file_name": self.images})
        table['label'] = table.file_name.apply(self.collect_labels)
        table.to_csv(os.path.join(self.src,"images", "ground_truth.csv"), index=False)


if __name__ == '__main__':
    src = os.path.join(os.pardir, 'data', 'breast', 'fea', 'official_test')
    OfficialFeaDataOrganizer(src).organize(ignore=True)
    src = os.path.join(os.pardir, 'data', 'breast', 'fea', 'official_train')
    OfficialFeaDataOrganizer(src).organize(ignore=False)
