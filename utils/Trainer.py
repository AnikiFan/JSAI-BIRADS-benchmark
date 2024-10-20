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
        self.loss, self.f1_score, self.accuracy, self.confusion_matrix = None, None, None, None
        self.train_transform = instantiate(self.cfg.train_transform)
        self.valid_transform = instantiate(self.cfg.valid_transform)
        # 用于记录以折为单位的最佳指标
        self.best_vloss, self.best_overall, = 1_000_000., -1

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
                self.loss, self.f1_score, self.accuracy, self.confusion_matrix = loss, f1_score, accuracy, confusion_matrix
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
            info(f'ACCURACY             :{self.accuracy:.10f}')
            info(f'F1                   :{self.f1_score:.10f}')
            info(f'OVERALL SCORE        :{self.accuracy *0.6+self.f1_score*0.4:.10f}')
            info(f'confusion matrix:\n{str(self.confusion_matrix)}')
        return instantiate(self.cfg.train.choose_strategy, loss=self.loss, accuracy=self.accuracy,
                           f1_score=self.f1_score)

    def train_one_epoch(self, *, model, train_loader: DataLoader, optimizer, epoch_index: int,
                        tb_writer: SummaryWriter) -> Tuple[float, float, float, torch.Tensor]:
        '''
        训练一个 epoch
        :param model: 模型
        :param epoch_index: 当前 epoch
        :param train_loader: 训练数据加载器
        :param num_class: 类别数量
        :param tb_writer: TensorBoard 写入器
        '''
        outputs, labels = [], []
        train_outputs, train_labels = [], []

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
                    or len(train_loader) < self.cfg.train.info_frequency and i == len(train_loader) - 1):
                # 计算实际的批次数
                train_outputs.extend(outputs)
                train_labels.extend(labels)
                outputs, labels = torch.cat(outputs, dim=0), torch.cat(labels, dim=0)
                avg_loss = self.loss_fn(input=outputs, target=labels).item()
                avg_accuracy = instantiate(self.cfg.train.accuracy, input=outputs, target=labels,num_classes=self.cfg.dataset.num_classes).item()
                avg_f1 = instantiate(self.cfg.train.f1_score, input=outputs, target=labels,num_classes=self.cfg.dataset.num_classes).item()
                tb_x = (epoch_index - 1) * len(train_loader) + i
                tb_writer.add_scalar('Loss/train', avg_loss, tb_x)
                tb_writer.add_scalar('Accuracy/train', avg_accuracy, tb_x)
                tb_writer.add_scalar('F1/train', avg_f1, tb_x)
                outputs, labels = [], []
        train_outputs.extend(outputs)
        train_labels.extend(labels)
        train_outputs = torch.cat(train_outputs, dim=0)
        train_labels = torch.cat(train_labels, dim=0)
        return (self.loss_fn(train_outputs,train_labels).item(),
                instantiate(self.cfg.train.accuracy, input=train_outputs, target=train_labels,
                            num_classes=self.cfg.dataset.num_classes).item(),
                instantiate(self.cfg.train.f1_score, input=train_outputs, target=train_labels,
                            num_classes=self.cfg.dataset.num_classes).item(),
                instantiate(self.cfg.train.confusion_matrix, input=train_outputs, target=train_labels,
                            num_classes=self.cfg.dataset.num_classes))

    def make_writer_title(self) -> str:
        """
        制作tensorboard的标题
        :return:
        """
        return '_'.join(map(str, [self.cfg.model._target_.split('.')[-1], self.cfg.model.lr,
                                  '-'.join(HydraConfig.get().runtime.output_dir.split(os.sep)[-2:]), self.cur_fold,
                                  'fold']))

    @staticmethod
    def init_weights(m):
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
            info('initialize model with xavier method')
            model.forward(next(iter(train_loader))[0].to(self.cfg.env.device))
            model.apply(Trainer.init_weights)
        optimizer = instantiate(self.cfg.optimizer, params=model.parameters())
        schedular = instantiate(self.cfg.schedular, optimizer=optimizer)
        writer = SummaryWriter(os.path.join('runs', self.make_writer_title()))
        # model.to(torch.device(self.cfg.env.device)) 初始化已经设置了device
        # 定义检查点路径
        early_stopping = instantiate(self.cfg.train.early_stopping)
        for epoch in range(1, self.cfg.train.epoch_num + 1):
            # 训练一个epoch，获取在上面的指标
            avg_loss, avg_accuracy, avg_f1, train_confusion_matrix = self.train_one_epoch(model=model,
                                                                                          train_loader=train_loader,
                                                                                          optimizer=optimizer,
                                                                                          epoch_index=epoch,
            
            # 更新学习率调度器
            if isinstance(self.schedular, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.schedular.step(avg_vloss)  # 使用验证损失作为指标
            else:
                self.schedular.step()
            model.eval()
            with torch.no_grad():
                valid_outcomes = [(vlabel.to(self.cfg.env.device), model(vinputs.to(self.cfg.env.device))) for
                                  vinputs, vlabel in tqdm(valid_loader, desc=f"Validating Epoch {epoch}", leave=False)]
            target = torch.cat([pair[0] for pair in valid_outcomes], dim=0)
            prediction = torch.cat([pair[1] for pair in valid_outcomes], dim=0)
            # 在该epoch的验证集上获得的指标
            avg_vloss = self.loss_fn(prediction, target).item()
            avg_vaccuracy = instantiate(self.cfg.train.accuracy, input=prediction.cpu(), target=target.cpu(),
                                        num_classes=self.cfg.dataset.num_classes).item()
            avg_vf1 = instantiate(self.cfg.train.f1_score, input=prediction.cpu(), target=target.cpu(), average='macro',
                                  num_classes=self.cfg.dataset.num_classes).item()
            confusion_matrix = instantiate(self.cfg.train.confusion_matrix, input=prediction.cpu(), target=target.cpu(),
                                           num_classes=self.cfg.dataset.num_classes)
            info(f"----------------- Epoch {epoch} Summary -----------------")
            info('LOSS           train {:.10f} valid {:.10f}'.format(avg_loss, avg_vloss))
            info('ACCURACY       train {:.10f} valid {:.10f}'.format(avg_accuracy, avg_vaccuracy))
            info('F1             train {:.10f} valid {:.10f}'.format(avg_f1, avg_vf1))
            info('OVERALL SCORE  train {:.10f} valid {:.10f}'.format(avg_accuracy*0.6+avg_f1*0.4, avg_vaccuracy*0.6+avg_vf1*0.4))
            info(f'confusion matrix train:\n{str(train_confusion_matrix)}')
            info(f'confusion matrix valid:\n{str(confusion_matrix)}')
            # Log the running loss averaged per batch
            # for both training and validation
            writer.add_scalars('Training vs. Validation Loss', {'Training': avg_loss, 'Validation': avg_vloss},
                               epoch)
            writer.add_scalars('Training vs. Validation Accuracy',
                               {'Training': avg_accuracy, 'Validation': avg_vaccuracy},
                               epoch)
            writer.add_scalars('Training vs. Validation F1', {'Training': avg_f1, 'Validation': avg_vf1},
                               epoch)
            writer.add_scalars('Training vs. Validation Overall', {'Training': avg_f1*0.4+avg_accuracy*0.6, 'Validation': avg_vf1*0.4+avg_vaccuracy*0.6},
                               epoch)
            writer.add_text(f'train confusion matrix of epoch {epoch}', str(train_confusion_matrix))
            writer.add_text(f'valid confusion matrix of epoch {epoch}', str(confusion_matrix))
            writer.flush()

            # 判断是否要更新在本折上获取的最佳指标
            if avg_vloss < best_loss:
                best_loss, best_f1, best_accuracy, best_confusion_matrix = avg_vloss, avg_vf1, avg_vaccuracy, confusion_matrix
            # 判断是否要更新在所有折上获取的最佳指标，若更新，同时保存参数
            if os.path.exists(os.path.join(HydraConfig.get().runtime.output_dir, 'last.pth')):
                os.remove(os.path.join(HydraConfig.get().runtime.output_dir, 'last.pth'))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': schedular.state_dict(),
                'best_vloss': self.best_vloss
            }, os.path.join(HydraConfig.get().runtime.output_dir, "last.pth"))
            if 0.6*avg_vaccuracy+0.4*avg_vf1 > self.best_overall:
                self.best_overall = 0.6*avg_vaccuracy+0.4*avg_vf1
                info(f"\n###############################################################\n"
                     f"# Validation overall improved to {0.6*avg_vaccuracy+0.4*avg_vf1:.6f} - saving best model #\n"
                     f"###############################################################")
                # 保存checkpoint（包括epoch，model_state_dict，optimizer_state_dict，best_vloss，但仅在best时保存）
                # 保存断点重训所需的信息（需要包括epoch，model_state_dict，optimizer_state_dict，best_vloss）
                # 只保留一份最佳指标对应的参数
                if os.path.exists(os.path.join(HydraConfig.get().runtime.output_dir, 'best_overall.pth')):
                    os.remove(os.path.join(HydraConfig.get().runtime.output_dir, 'best_overall.pth'))
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': schedular.state_dict(),
                    'best_vloss': self.best_vloss
                }, os.path.join(HydraConfig.get().runtime.output_dir, "best_overall.pth"))
            if avg_vloss < self.best_vloss:
                self.best_vloss = avg_vloss
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
                }, os.path.join(HydraConfig.get().runtime.output_dir, "best_vloss.pth"))
            early_stopping(train_loss=avg_loss,val_loss=avg_vloss)
            # 检查早停条件
            if early_stopping.early_stop:
                info("Early stopping triggered!")
                break
        
        # env=zhy_remote且设置为final时，自动释放远程服务器。
        # if cfg.env.final:
        #     print("final complete, release remote server")
        #     os.system("export $(cat /proc/1/environ |tr '\\0' '\\n' | grep MATCLOUD_CANCELTOKEN)&&/public/script/matncli node cancel -url https://matpool.com/api/public/node")
        return best_loss, best_f1, best_accuracy, best_confusion_matrix
