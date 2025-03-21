import pandas as pd
import numpy as np
import os
from config.config import init_config, Config
from fontTools.varLib.instancer import instantiateSTAT
from hydra import main
from hydra.utils import instantiate
import torch
from utils.TableDataset import TableDataset
from tqdm import tqdm
from omegaconf import OmegaConf
from torch.utils.data import DataLoader


class Tester:
    def __init__(self, task, cfg: Config, data_folder_path: str, check_point_folder_path: str):
        assert task in ['cla', 'fea'], "task should be either cla or fea!"
        assert os.path.exists(
            os.path.join(data_folder_path, os.pardir, 'docs', 'cla_order.csv')), "cla_order.csv doesn't exist!"
        assert os.path.exists(
            os.path.join(data_folder_path, os.pardir, 'docs', 'fea_order.csv')), "fea_order.csv doesn't exist!"
        self.data_folder_path = data_folder_path
        self.cfg = cfg
        self.task = task
        self.check_point_folder_path = check_point_folder_path
        self.ground_truth = pd.read_csv(os.path.join(data_folder_path, 'breast', task, 'test', 'ground_truth.csv'),
                                        dtype=str)
        # 这里images_name只是为了排序用，不能用于读取文件，因为像是0086变为了86
        # 官方提供的cla_order中，列名为images_name，而fea_order中，列名为image_name
        self.image_name_key = 'image_name' if task == 'fea' else 'images_name'
        self.ground_truth[self.image_name_key] = self.ground_truth.file_name.str.split('.').str[0].apply(
            lambda x: str(int(x)) if x.isnumeric() else x)
        self.ground_truth.set_index(keys=self.image_name_key, inplace=True, drop=True)
        self.order = pd.read_csv(os.path.join(data_folder_path, os.pardir, 'docs', task + '_order.csv'))
        # 按照cla_order进行重排序
        self.ground_truth = self.ground_truth.loc[self.order[self.image_name_key], :]
        self.ground_truth.reset_index(drop=True, inplace=True)
        self.ground_truth['id'] = np.arange(1, len(self.ground_truth) + 1, 1)
        self.ground_truth.set_index(keys="id", inplace=True, drop=True)
        if task == 'fea':
            self.ground_truth['boundary'] = self.ground_truth.label.str[0]
            self.ground_truth['calcification'] = self.ground_truth.label.str[1]
            self.ground_truth['direction'] = self.ground_truth.label.str[2]
            self.ground_truth['shape'] = self.ground_truth.label.str[3]
        self.pre = self.ground_truth.copy()
        if task == 'fea':
            self.pre.boundary = -1
            self.pre.calcification = -1
            self.pre.direction = -1
            self.pre.shape = -1
        else:
            self.pre.label = -1
        self.ground_truth.file_name = self.ground_truth.file_name.apply(
            lambda x: os.path.join(self.data_folder_path, 'breast', task, 'test', x))
        self.data_loader = DataLoader(TableDataset(self.ground_truth[["file_name", 'label']], image_format='Tensor',
                                                   transform=instantiate(self.cfg.valid_transform)), shuffle=False, batch_size=1)

    def test(self):
        checkpoint = torch.load(os.path.join(self.check_point_folder_path, "model.pth"), weights_only=True)
        model = instantiate(self.cfg.model, num_classes=self.cfg.dataset.num_classes,
                            model_weight_path=self.cfg.env.model_weight_path).to(self.cfg.env.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        id = 1
        if self.task == 'cla':
            for image, label in tqdm(self.data_loader):
                self.pre.loc[id, "label"] = model(image.to(self.cfg.env.device)).argmax().item()
                id += 1
        else:
            for image, label in tqdm(self.data_loader):
                result = torch.where(model(image.to(self.cfg.env.device)) < 0, 0, 1).flatten()
                self.pre.loc[id, "boundary"] = result[0].item()
                self.pre.loc[id, "calcification"] = result[1].item()
                self.pre.loc[id, "direction"] = result[2].item()
                self.pre.loc[id, "shape"] = result[3].item()
                id += 1
        os.makedirs(os.path.join(self.check_point_folder_path, "submit"), exist_ok=True)
        if self.task == 'fea':
            self.ground_truth.reset_index(drop=False).drop(['file_name', 'label'], axis=1).to_csv(
                os.path.join(self.check_point_folder_path, "submit", self.task + "_gt.csv"), index=False)
        else:
            self.ground_truth.reset_index(drop=False).drop(['file_name'], axis=1).to_csv(
                os.path.join(self.check_point_folder_path, "submit", self.task + "_gt.csv"), index=False)
        self.pre.reset_index(drop=False).drop(['file_name', 'label'], axis=1).to_csv(
            os.path.join(self.check_point_folder_path, "submit", self.task + "_pre.csv"),
            index=False)
        self.order.loc[:, ['id', self.image_name_key]].to_csv(
            os.path.join(self.check_point_folder_path, "submit", self.task + "_order.csv"), index=False, )


checkpoint_path = os.path.join(os.curdir, "outputs", "2024-10-09", "10-11-33")
task = 'cla'


@main(version_base=None, config_name="config", config_path=os.path.join(checkpoint_path, ".hydra"))
def test(cfg: Config):
    print(OmegaConf.to_yaml(cfg))
    Tester(task, cfg, os.path.join(os.curdir, "data"), checkpoint_path).test()


init_config()
test()
