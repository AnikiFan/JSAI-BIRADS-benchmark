import pandas as pd
from logging import warning
import os
from tqdm import tqdm
import shutil


class OfficialClaDataOrganizer:
    """
    针对官方Cla数据集进行整理，即按照2类，3类……文件夹进行分类的数据集
    同时带有筛出不符合规范的label.txt的功能
    用于汇总图像至一个文件夹中，并形成ground_truth.csv
    """

    def __init__(self, src:str, dst:str):
        """
        注意，这里按照字典序为各个文件夹中的图像赋予label，例如2类赋0，3类赋1，4a类赋2
        :param src: 2类，3类……文件夹所在目录
        :param dst: 整理后存储的目录
        """
        self.src = src
        self.dst = dst

    @staticmethod
    def check_label(ground_truth:int, base_path:str, file_name:str)->bool:
        """
        给定样本名称，判断文本标签是否合法，这里需要满足1.只有一行；2.有五个用空格相隔的数字；3.与文件夹对应的标签一致
        :param ground_truth:
        :param base_path:
        :param file_name:
        :return:
        """
        with open(os.path.join(base_path, 'labels', file_name), 'r') as file:
            content = file.readlines()
            if len(content) != 1:
                warning(f"invalid label num:{os.path.join(base_path, 'labels', file_name)}")
                return False
            if len(content[0].split()) != 5:
                warning(f"invalid label:{os.path.join(base_path, 'labels', file_name)}")
                return False
            if content[0].split()[0] != str(ground_truth):
                warning(f"incompatible label:{os.path.join(base_path, 'labels', file_name)}")
                return False
        return True

    @staticmethod
    def move_image(image_path:str, label_name:str, dst:str)->None:
        """
        给定图像名称，不包含后缀名，将对应的jpg和png格式复制到指定文件夹中
        :param image_path:
        :param label_name:
        :param dst:
        :return:
        """
        png_suffix = label_name.replace('txt', 'png')
        jpg_suffix = label_name.replace('txt', 'jpg')
        if os.path.exists(os.path.join(image_path, png_suffix)):
            shutil.copy(os.path.join(image_path, png_suffix), dst)
        elif os.path.exists(os.path.join(image_path, jpg_suffix)):
            shutil.copy(os.path.join(image_path, jpg_suffix), dst)
        else:
            warning(f"image {image_path + label_name.replace('.txt')} doesn't exist!")

    def organize(self, ignore:bool)->None:
        """
        各文件夹的label按照文件夹名称字典序赋值，从0开始
        以合法的txt标签文件来进一步搜索对应图片
        :param ignore: 是否要检验txt标签文件的合法性
        :return:
        """
        if os.path.exists(self.dst):
            shutil.rmtree(self.dst)
        os.makedirs(self.dst)
        label_tables = []
        
        folders = os.listdir(self.src)
        legal_folders = ["2类", "3类", "4A类", "4B类", "4C类", "5类"]
        folders = list(filter(lambda x: x in legal_folders, folders))
        assert len(folders) == len(legal_folders), "folders not match! current folders:{}".format(folders) # 检查是否包含所有合法文件夹
        
        for label, folder in tqdm(enumerate(folders), total=len(folders)):
            if ignore:
                valid_labels = os.listdir(os.path.join(self.src, folder, 'labels'))
            if not ignore:
                valid_labels = list(
                    filter(lambda x: OfficialClaDataOrganizer.check_label(label + 1, os.path.join(self.src, folder), x),
                           os.listdir(os.path.join(self.src, folder, 'labels'))))
            list(map(lambda x: OfficialClaDataOrganizer.move_image(os.path.join(self.src, folder, 'images'), x,
                                                                   self.dst), valid_labels))
            file_names = list(map(lambda x: x.replace('txt', 'png') if os.path.exists(
                os.path.join(self.dst, x.replace('txt', 'png'))) else x.replace('txt', 'jpg'), valid_labels))
            label_table = pd.DataFrame({'file_name': file_names})
            label_table['label'] = label
            label_tables.append(label_table)
        out = pd.concat(label_tables, axis=0).reset_index(drop=True)
        out.columns = ['file_name', 'label']
        out.to_csv(os.path.join(self.dst, 'ground_truth.csv'), index=False)


if __name__ == '__main__':
    pre_dir = os.path.join(os.getcwd(), 'data', 'breast', 'cla')
    
    src = os.path.join(pre_dir, 'official_test')
    dst = os.path.join(pre_dir, 'test')
    OfficialClaDataOrganizer(src, dst).organize(ignore=True)
    src = os.path.join(pre_dir, 'official_train')
    dst = os.path.join(pre_dir, 'train')
    OfficialClaDataOrganizer(src, dst).organize(ignore=False)
