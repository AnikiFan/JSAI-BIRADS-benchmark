import shutil
import numpy as np
from tqdm import tqdm
import torch
from torcheval.metrics.functional import multiclass_f1_score, multiclass_accuracy, multiclass_confusion_matrix
from config.config import Config
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from logging import info
import os
from utils.tools import save_checkpoint
from torch.utils.tensorboard import SummaryWriter
from utils.BreastDataset import BreastCrossValidationData, getBreastTrainValidData
from torch.utils.data import DataLoader
from utils.time_logger import time_logger
from typing import *
from hydra import main


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg: Config = cfg
        self.loss_fn = instantiate(cfg.train.loss_function)
        self.cur_fold = 1
        # 用于统计各折之间的指标
        self.loss, self.f1_score, self.accuracy, self.confusion_matrix = None,None,None,None
        self.train_transform = instantiate(self.cfg.train_transform)
        self.valid_transform = instantiate(self.cfg.valid_transform)
        # 用于记录以折为单位的最佳指标
        self.best_vloss, self.best_vf1, self.best_vaccuracy, self.best_vconfusion_matrix = 1_000_000., None, None, None

    def train(self) -> None:
        for train_ds, valid_ds in instantiate(self.cfg.dataset, data_folder_path=self.cfg.env.data_folder_path,
                                              train_transform=self.train_transform,
                                              valid_transform=self.valid_transform):
            loss, f1_score, accuracy, confusion_matrix = self.train_one_fold(
                DataLoader(train_ds, batch_size=self.cfg.train.batch_size, shuffle=True,
                           pin_memory=self.cfg.env.pin_memory,
                           drop_last=False, num_workers=self.cfg.train.num_workers,
                           pin_memory_device=self.cfg.env.pin_memory_device),
                DataLoader(valid_ds, batch_size=self.cfg.train.batch_size, shuffle=False,
                           pin_memory=self.cfg.env.pin_memory,
                           drop_last=False, num_workers=self.cfg.train.num_workers,
                           pin_memory_device=self.cfg.env.pin_memory_device)
            )
            if self.loss is None:
                self.loss,self.f1_score,self.accuracy,self.confusion_matrix = loss,f1_score,accuracy,confusion_matrix
            else:
                self.loss += loss
                self.f1_score += f1_score
                self.accuracy += accuracy
                self.confusion_matrix += confusion_matrix
            self.cur_fold += 1
        fold_num = self.cur_fold - 1
        if fold_num != 1:
            self.loss /= fold_num
            self.f1_score /= fold_num
            self.accuracy /= fold_num
            info(f"***************** {fold_num} folds' summary *****************")
            info(f'LOSS                 :{self.loss:.10f}')
            info(f'ACCURACY             :{self.f1_score:.10f}')
            info(f'F1                   :{self.accuracy:.10f}')
            info(f'confusion matrix:\n{str(self.confusion_matrix)}')
        return instantiate(self.cfg.train.choose_strategy, loss=self.loss, accuracy=self.accuracy,
                           f1_score=self.f1_score)

    def train_one_epoch(self, *, model, train_loader: DataLoader, optimizer, epoch_index: int,
                        tb_writer: SummaryWriter) -> Tuple[float, float, float]:
        '''
        训练一个 epoch
        :param model: 模型
        :param epoch_index: 当前 epoch
        :param train_loader: 训练数据加载器
        :param num_class: 类别数量
        :param tb_writer: TensorBoard 写入器
        '''
        outputs, labels = [], []

        model.train(True)

        for i, data in enumerate(tqdm(train_loader, desc=f"Training   Epoch {epoch_index}", leave=False), start=1):
            input, label = data
            input = input.to(self.cfg.env.device)
            label = label.to(self.cfg.env.device)
            optimizer.zero_grad()
            output = model(input)
            outputs.append(output)
            labels.append(label)
            loss = self.loss_fn(output, label)
            loss.backward()
            optimizer.step()
            if (i % self.cfg.train.info_frequency == 0
                    or len(train_loader) > self.cfg.train.info_frequency and i == len(train_loader) - 1):
                # 计算实际的批次数
                outputs, labels = torch.cat(outputs, dim=0), torch.cat(labels, dim=0)
                avg_loss = self.loss_fn(input=outputs, target=labels).item()
                avg_f1 = instantiate(self.cfg.train.f1_score, input=outputs, target=labels,
                                     num_classes=self.cfg.dataset.num_classes).item()
                avg_accuracy = instantiate(self.cfg.train.accuracy, input=outputs, target=labels,
                                           num_classes=self.cfg.dataset.num_classes).item()
                tb_x = (epoch_index - 1) * len(train_loader) + i
                tb_writer.add_scalar('Loss/train', avg_loss, tb_x)
                tb_writer.add_scalar('Accuracy/train', avg_accuracy, tb_x)
                tb_writer.add_scalar('F1/train', avg_f1, tb_x)
                outputs, labels = [], []
        # 为了避免指标出现大幅波动，不对尾部剩余的一小部分计算指标
        return avg_loss, avg_accuracy, avg_f1

    def make_writer_title(self) -> str:
        """
        制作tensorboard的标题
        :return:
        """
        return '_'.join(map(str, [self.cfg.model._target_, self.cfg.model.lr,
                                  '-'.join(HydraConfig.get().runtime.output_dir.split(os.sep)[-2:]), self.cur_fold,
                                  'fold']))

    @staticmethod
    def init_weights(m):
        """
        使用xavier方法初始化权重
        :param m:
        :return:
        """
        if type(m) == torch.nn.Linear or type(m) == torch.nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)

    @time_logger
    def train_one_fold(self, train_loader: DataLoader, valid_loader: DataLoader) -> Tuple[
        float, float, float, torch.Tensor]:
        """
        训练一折
        :param train_loader:
        :param valid_loader:
        :return: 该折训练中，在单个验证集上达到的最佳的指标
        """
        # 用于计算以epoch为单位的最佳指标
        best_loss, best_f1, best_accuracy, best_confusion_matrix = 1_000_000., None, None, None
        model = instantiate(self.cfg.model, num_classes=self.cfg.dataset.num_classes,
                            model_weight_path=self.cfg.env.model_weight_path).to(self.cfg.env.device)
        # 为了初始化lazy layer，先传入一张图片
        # 初始化权重
        if not self.cfg.model.pretrained:
            model.forward(next(iter(train_loader))[0].to(self.cfg.env.device))
            model.apply(Trainer.init_weights)
        optimizer = instantiate(self.cfg.optimizer, params=model.parameters())
        schedular = instantiate(self.cfg.schedular, optimizer=optimizer)
        writer = SummaryWriter(os.path.join('runs', self.make_writer_title()))
        model.to(torch.device(self.cfg.env.device))
        # 定义检查点路径
        early_stopping = instantiate(self.cfg.train.early_stopping)
        for epoch in range(1, self.cfg.train.epoch_num + 1):
            # 训练一个epoch，获取在上面的指标
            avg_loss, avg_accuracy, avg_f1 = self.train_one_epoch(model=model, train_loader=train_loader,
                                                                  optimizer=optimizer, epoch_index=epoch,
                                                                  tb_writer=writer)
            schedular.step()
            model.eval()
            with torch.no_grad():
                valid_outcomes = [(vlabel.to(self.cfg.env.device), model(vinputs.to(self.cfg.env.device))) for
                                  vinputs, vlabel in tqdm(valid_loader, desc=f"Validating Epoch {epoch}", leave=False)]
            target = torch.cat([pair[0] for pair in valid_outcomes], dim=0)
            prediction = torch.cat([pair[1] for pair in valid_outcomes], dim=0)
            # 在该epoch的验证集上获得的指标
            avg_vloss = self.loss_fn(prediction, target).item()
            avg_vaccuracy = instantiate(self.cfg.train.accuracy, input=prediction, target=target,
                                        num_classes=self.cfg.dataset.num_classes).item()
            avg_vf1 = instantiate(self.cfg.train.f1_score, input=prediction, target=target, average='macro',
                                  num_classes=self.cfg.dataset.num_classes).item()
            confusion_matrix = instantiate(self.cfg.train.confusion_matrix, input=prediction, target=target,
                                           num_classes=self.cfg.dataset.num_classes)
            info(f"----------------- Epoch {epoch} Summary -----------------")
            info('LOSS      train {:.10f} valid {:.10f}'.format(avg_loss, avg_vloss))
            info('ACCURACY  train {:.10f} valid {:.10f}'.format(avg_accuracy, avg_vaccuracy))
            info('F1        train {:.10f} valid {:.10f}'.format(avg_f1, avg_vf1))
            info(f'confusion matrix:\n{str(confusion_matrix)}')
            # Log the running loss averaged per batch
            # for both training and validation
            writer.add_scalars('Training vs. Validation Loss', {'Training': avg_loss, 'Validation': avg_vloss},
                               epoch)
            writer.add_scalars('Training vs. Validation Accuracy',
                               {'Training': avg_accuracy, 'Validation': avg_vaccuracy},
                               epoch)
            writer.add_scalars('Training vs. Validation F1', {'Training': avg_f1, 'Validation': avg_vf1},
                               epoch)
            writer.add_text(f'confusion matrix of epoch {epoch}', str(confusion_matrix))
            writer.flush()

            # 判断是否要更新在本折上获取的最佳指标
            if avg_vloss < best_loss:
                best_loss, best_f1, best_accuracy, best_confusion_matrix = avg_vloss, avg_vf1, avg_vaccuracy, confusion_matrix
            # 判断是否要更新在所有折上获取的最佳指标，若更新，同时保存参数
            if avg_vloss < self.best_vloss:
                self.best_vloss, self.best_vf1, self.best_vaccuracy, self.best_vconfusion_matrix = avg_vloss, avg_vf1, avg_vaccuracy, confusion_matrix
                info(f"\n############################################################\n"
                     f"# Validation loss improved to {avg_vloss:.6f} - saving best model #\n"
                     f"############################################################")
                # 保存checkpoint（包括epoch，model_state_dict，optimizer_state_dict，best_vloss，但仅在best时保存）
                # 保存断点重训所需的信息（需要包括epoch，model_state_dict，optimizer_state_dict，best_vloss）
                # 只保留一份最佳指标对应的参数
                if os.path.exists(os.path.join(HydraConfig.get().runtime.output_dir, 'model.pth')):
                    os.remove(os.path.join(HydraConfig.get().runtime.output_dir, 'model.pth'))
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': schedular.state_dict(),
                    'best_vloss': self.best_vloss
                }, os.path.join(HydraConfig.get().runtime.output_dir, "model.pth"))
            early_stopping(avg_vloss)
            # 检查早停条件
            if early_stopping.early_stop:
                info("Early stopping triggered!")
                break
        return best_loss, best_f1, best_accuracy, best_confusion_matrix
