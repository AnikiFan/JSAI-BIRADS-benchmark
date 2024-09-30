import pandas as pd
import numpy as np
import os


class Tester:
    num_cla_testA = 671
    def __init__(self, data_folder_path):
        """
        实例化该类之前，需要先运行OfficialClaDataOrganizer.py
        并将cla_order.csv放入test文件夹中
        :param data_folder_path:
        """
        assert os.path.exists(
            os.path.join(data_folder_path, 'breast', 'cla', 'test')), "请先运行OfficialClaDataOrganizer.py"
        assert os.path.exists(os.path.join(data_folder_path, 'breast', 'cla', 'test',
                                           'cla_order.csv')), f"请将cla_order.csv放入{os.path.join(data_folder_path, 'breast', 'cla', 'test', 'cla_order.csv')}"
        assert len(os.listdir(os.path.join(data_folder_path, 'breast', 'cla',
                                           'test'))) == self.num_cla_testA+2, "文件数量不正确，请以ignore=True运行OfficialClaDataOrganizer.py"
        self.ground_truth = pd.read_csv(os.path.join(data_folder_path, 'breast', 'cla', 'test','ground_truth.csv'))
        # 这里images_name只是为了排序用，不能用于读取文件，因为像是0086变为了86
        self.ground_truth['images_name'] = self.ground_truth.file_name.str.split('.').str[0].apply(lambda x:str(int(x)) if x.isnumeric() else x)
        self.ground_truth.set_index(keys="images_name",inplace=True,drop=True)
        self.cla_order = pd.read_csv(os.path.join(data_folder_path, 'breast', 'cla', 'test','cla_order.csv'))
        # 按照cla_order进行重排序
        self.ground_truth = self.ground_truth.loc[self.cla_order.images_name,:]
        self.ground_truth.reset_index(drop=True, inplace=True)
        self.ground_truth['id'] = np.arange(1,len(self.ground_truth)+1,1)
        self.ground_truth.set_index(keys="id",inplace=True,drop=True)
        self.cla_pre = self.ground_truth.copy()
        self.cla_pre.label = -1
        pass


if __name__ == '__main__':
    Tester(os.path.join(os.pardir,'data'))