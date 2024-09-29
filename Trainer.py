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
from utils.ClaDataset import ClaCrossValidationData, getClaTrainValidData
from torch.utils.data import DataLoader
from utils.time_logger import time_logger
from typing import *
from hydra import main


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg: Config = cfg
        self.loss_fn = instantiate(cfg.train.loss_function)
        self.cur_fold = 1
        self.loss, self.f1_score, self.accuracy, self.confusion_matrix = 0, 0, 0, torch.zeros(
            (self.cfg.train.num_classes, self.cfg.train.num_classes), dtype=torch.int, device=self.cfg.env.device)
        self.train_transform = instantiate(self.cfg.train_transform)
        self.valid_transform = instantiate(self.cfg.valid_transform)
        self.best_vloss, self.best_vf1, self.best_vaccuracy, self.best_vconfusion_matrix = 1_000_000., None, None, None

    def train(self):
        for train_ds, valid_ds in instantiate(self.cfg.dataset, data_folder_path=self.cfg.env.data_folder_path,
                                              train_transform=self.train_transform,
                                              valid_transform=self.valid_transform):
            loss, f1_score, accuracy, confusion_matrix = self.train_one_fold(
                DataLoader(train_ds, batch_size=self.cfg.train.batch_size, shuffle=True, pin_memory=True,
                           drop_last=False, num_workers=self.cfg.train.num_workers,
                           pin_memory_device=self.cfg.env.device),
                DataLoader(valid_ds, batch_size=self.cfg.train.batch_size, shuffle=True, pin_memory=True,
                           drop_last=False, num_workers=self.cfg.train.num_workers,
                           pin_memory_device=self.cfg.env.device)
            )
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
            info(f"\n-----------------{fold_num} folds' summary-----------------")
            info(f'LOSS                 :{self.loss}')
            info(f'ACCURACY             :{self.f1_score}')
            info(f'F1                   :{self.accuracy}')
            info(f'confusion matrix:\n{str(self.confusion_matrix)}')

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

        for i, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch_index}"), start=1):
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
            if i % self.cfg.train.info_frequency == 0:
                # 计算实际的批次数
                outputs, labels = torch.cat(outputs, dim=0), torch.cat(labels, dim=0)
                loss = self.loss_fn(input=outputs, target=labels) / self.cfg.train.info_frequency
                f1 = multiclass_f1_score(input=outputs, target=labels) / self.cfg.train.info_frequency
                accuracy = multiclass_accuracy(input=outputs, target=labels) / self.cfg.train.info_frequency

                tb_x = epoch_index * len(train_loader) + i
                tb_writer.add_scalar('Loss/train', loss, tb_x)
                tb_writer.add_scalar('Accuracy/train', accuracy, tb_x)
                tb_writer.add_scalar('F1/train', f1, tb_x)
                outputs, labels = [], []
        # 为了避免指标出现大幅波动，不对尾部剩余的一小部分计算指标
        return loss, accuracy, f1

    def make_writer_title(self) -> str:
        """
        制作tensorboard的标题
        :return:
        """
        return '_'.join(map(str, [self.cfg.model._target_, self.cfg.optimizer.lr,
                                  '-'.join(HydraConfig.get().runtime.output_dir.split(os.sep)[-2:]), self.cur_fold,
                                  'fold']))

    @time_logger
    def train_one_fold(self, train_loader: DataLoader, valid_loader: DataLoader) -> Tuple[
        float, float, float, torch.Tensor]:
        """
        训练一折
        :param train_loader:
        :param valid_loader:
        :return: 该折训练中，在单个验证集上达到的最佳的指标
        """
        best_loss, best_f1, best_accuracy, best_confusion_matrix = 1_000_000., None, None, None
        model = instantiate(self.cfg.model)
        optimizer = instantiate(self.cfg.optimizer, params=model.parameters())
        writer = SummaryWriter(os.path.join('runs', self.make_writer_title()))

        model.to(torch.device(self.cfg.env.device))
        epoch_number = 0
        # 定义检查点路径
        checkPoint_path = HydraConfig.get().runtime.output_dir
        early_stopping = instantiate(self.cfg.train.early_stopping)
        for epoch in range(1, self.cfg.train.epoch_num + 1):
            # 训练一个epoch，获取在上面的指标
            avg_loss, avg_accuracy, avg_f1 = self.train_one_epoch(model=model, train_loader=train_loader,
                                                                  optimizer=optimizer, epoch_index=epoch,
                                                                  tb_writer=writer)
            model.eval()
            with torch.no_grad():
                valid_outcomes = [(vlabel.to(self.cfg.env.device), model(vinputs.to(self.cfg.env.device))) for
                                  vinputs, vlabel in valid_loader]

            ground_truth = torch.cat([pair[0] for pair in valid_outcomes], dim=0)
            prediction = torch.cat([pair[1] for pair in valid_outcomes], dim=0)
            avg_vloss = self.loss_fn(prediction, ground_truth)
            avg_vaccuracy = multiclass_accuracy(input=prediction, target=ground_truth).tolist()
            avg_vf1 = multiclass_f1_score(input=prediction, target=ground_truth, average='macro',
                                          num_classes=self.cfg.train.num_classes).tolist()
            confusion_matrix = multiclass_confusion_matrix(input=prediction, target=ground_truth,
                                                           num_classes=self.cfg.train.num_classes)
            info('LOSS      train {} valid {}'.format(avg_loss, avg_vloss))
            info('ACCURACY  train {} valid {}'.format(avg_accuracy, avg_vaccuracy))
            info('F1        train {} valid {}'.format(avg_f1, avg_vf1))
            info(f'confusion matrix:\n{str(confusion_matrix)}')
            # Log the running loss averaged per batch
            # for both training and validation
            writer.add_scalars('Training vs. Validation Loss', {'Training': avg_loss, 'Validation': avg_vloss},
                               epoch_number)
            writer.add_scalars('Training vs. Validation Accuracy',
                               {'Training': avg_accuracy, 'Validation': avg_vaccuracy},
                               epoch_number)
            writer.add_scalars('Training vs. Validation F1', {'Training': avg_f1, 'Validation': avg_vf1},
                               epoch_number)
            writer.add_text(f'confusion matrix of epoch {epoch_number}', str(confusion_matrix))
            writer.flush()

            # 检查早停条件
            if avg_vloss < best_loss:
                best_loss, best_f1, best_accuracy, best_confusion_matrix = avg_vloss, avg_vf1, avg_vaccuracy, confusion_matrix
            if avg_vloss < self.best_vloss:
                self.best_vloss, self.best_vf1, self.best_vaccuracy, self.best_vconfusion_matrix = avg_vloss, avg_vf1, avg_vaccuracy, confusion_matrix
                info(f"\n=> Validation loss improved to {avg_vloss:.6f} - saving best model\n")

                # 保存checkpoint（包括epoch，model_state_dict，optimizer_state_dict，best_vloss，但仅在best时保存）
                # 保存断点重训所需的信息（需要包括epoch，model_state_dict，optimizer_state_dict，best_vloss）
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_vloss': self.best_vloss
                }
                if os.path.exists(os.path.join(checkPoint_path, 'resume_checkpoint')):
                    shutil.rmtree(os.path.join(checkPoint_path, 'resume_checkpoint'))
                os.makedirs(os.path.join(checkPoint_path, 'resume_checkpoint'))
                save_checkpoint(checkpoint, checkPoint_path, filename=os.path.
                                join('resume_checkpoint', f'epoch{epoch}_vloss{avg_vloss:.4f}_f1{avg_vf1:.4f}.pth'))

            early_stopping(avg_vloss)
            if early_stopping.early_stop:
                info("Early stopping triggered!")
                break

            epoch_number += 1

        return best_loss, best_f1, best_accuracy, best_confusion_matrix
