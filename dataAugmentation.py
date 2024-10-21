from utils.dataAugmentation import Preprocess, MixUp
import albumentations as A
import numpy as np
from utils.classDistribution import getClassDistribution
import os
from typing import Literal, Optional
import logging
from logging import info,debug,warning
from config.config import Config



def calculateForAugmentation(mode: Literal['balance', 'same'] = 'balance',
                    target_num: Optional[np.ndarray] = None,
                    dataset: Optional[dict] = None):
    """
    计算数据增强所需的比率。
    参数:
    mode (str): 计算模式，可选 'balance' 或 'same'。
        - 'balance': 平衡模式，使每个类别的样本数“达到”目标数量。
        - 'same': 相同模式，使每个类别的样本数”增加“目标数量。
    target_num (np.array): 每个类别的目标样本数量，默认为 [500, 500, 500, 500, 500, 500]。
    dataset (dict): 指定要使用的数据集，包括以下键：
        - 'official_train': 是否使用官方训练集
        - 'BUS': 是否使用 BUS 数据集
        - 'USG': 是否使用 USG 数据集
        - 'trainROI': 是否使用 trainROI 数据集

    返回:
    ratio: 每个类别的增强比率。
    actual_target_num: 每个类别的实际增强数量。
    class_distribution: 每个类别的原始数量。

    注意:
    - 函数假设有6个类别。
    - 如果某个类别的目标数量小于当前数量，在 'balance' 模式下可能会得到负值比率。
    """
    assert mode in ['balance', 'same'], "mode must be 'balance' or 'same'"
    if target_num is None:
        target_num = np.array([500, 500, 500, 500, 500, 500])
    if dataset is None:
        dataset = {
            'official_train': False,
            'BUS': False,
            'USG': False,
            'trainROI': True,
        }

    class_distribution = np.array([0, 0, 0, 0, 0, 0])
    for dataset_name, is_used in dataset.items():
        if is_used:
            path = os.path.join(os.curdir, "data", 'breast', 'cla', dataset_name)
            _, class_distribution_list = getClassDistribution(path)
            class_distribution += np.array(class_distribution_list)
            
    if mode == 'balance':
        assert np.all(target_num >= class_distribution), "目标数量不能小于当前数量"
        actual_target_num = target_num - class_distribution
    elif mode == 'same':
        actual_target_num = target_num

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.divide(actual_target_num, class_distribution)
        ratio[~np.isfinite(ratio)] = 0  # 将 inf 和 nan 替换为0

    return ratio,actual_target_num,class_distribution


if __name__ == '__main__':
    dataset = {
        'official_train':False,
        'BUS':False,
        'USG':False,
        'trainROI':True,
    }
    target_num = np.array([500, 500, 500, 500, 500, 500])
    ratio,actual_target_num,class_distribution  = calculateForAugmentation(mode='same', target_num=target_num, dataset=dataset)
        # 设置要使用的增广策略
    selected_transforms = [
        # A.Rotate(limit=10, p=1.0),
        # A.HorizontalFlip(p=1.0),
        # A.VerticalFlip(p=1.0),
        # A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
        
        A.Compose([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
        ]),
        
        # A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
        A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
        ]),
        
        A.Compose([
            A.ElasticTransform(alpha=1.0, sigma=50, alpha_affine=50, p=1.0),
        ]),
        
        # A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        # A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=1.0)
        
        A.Compose([
            A.Rotate(limit=10, p=1.0)
        ]),
        # A.HorizontalFlip(p=1.0), 
        # A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
        # A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
        # A.ElasticTransform(alpha=1.0, sigma=50, alpha_affine=50, p=1.0),
        # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=0, p=1.0),
        # A.RandomGamma(gamma_limit=(80, 120), p=1.0)
        A.Compose([
            A.Rotate(limit=10, p=0.5),
            A.HorizontalFlip(p=0.5), 
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.ElasticTransform(alpha=1.0, sigma=50, alpha_affine=50, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=0, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5)
        ])
    ]
 
    
    mixup_ratio = ratio
    
    mixups = [
        'mixup_0.1',
    ]
    
 
 
 
    print("-"*100)
    print("增强信息如下")
    print(f"所使用数据集: {[k for k, v in dataset.items() if v]}")
    print(f"原始数据分布: {' '.join(f'{num:6d}' for num in class_distribution)}")
    print(f"各类增强数量: {' '.join(f'{num:6d}' for num in actual_target_num)}")
    print(f"各类增强比率: {' '.join(f'{ratio:6.3f}' for ratio in ratio)}")
    print("-"*100)
    input("检查完毕后按回车继续，ctrl+c退出:")
    print("-"*100)
    print("选择的数据增强策略:")
    for i, transform in enumerate(selected_transforms, 1):
        transforms = transform.transforms
        print(f"{i:2d}.")
        for transform in transforms:
            print(f"   {transform}")
    print(f"MixUp 策略:")
    for i, mixup in enumerate(mixups, 1):
        print(f"{i:2d}.")
        print(f"   {mixup}")
    print(f"-"*100)
    input("请检查以上增广信息。按回车键继续，或按 Ctrl+C 退出:")
    
    '''
    开始数据增强
    '''
    ratio = tuple(ratio)
    mixup_ratio = tuple(mixup_ratio)
    for transform in selected_transforms:
        print(f"增强策略: {transform}")
        Preprocess(transform, ratio=ratio, **dataset).process_image()
    
    for mixup_name in mixups:
        print(f"MixUp 策略: {mixup_name}")
        alpha = float(mixup_name.split('_')[1])  # 提取 MixUp 的 alpha 值
        MixUp(alpha, ratio=mixup_ratio, **dataset).process_image()
    
    

