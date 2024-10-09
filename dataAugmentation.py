from utils.dataAugmentation import Preprocess, MixUp
import albumentations as A
import numpy as np
# from debug import debug

'''
原始数据集
1    1061
0     849
2     404
3     274
5     269
4     232
'''


    

if __name__ == '__main__':
    # balanced_num = 500
    target_num = np.array([1000, 1000, 1000, 1000, 1000, 1000])
    ratio = target_num / np.array([1061, 849, 404, 274, 269, 232])
    # 将 ratio 转换为普通列表
    ratio = ratio.tolist()
    
    print(ratio)
    
    # 设置要使用的增广策略
    selected_transforms = [
        A.Rotate(limit=15, p=1.0),
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=1.0),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
        A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
        A.ElasticTransform(alpha=1.0, sigma=50, p=1.0),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=1.0)
    ]
    
    # 转化成compose
    selected_transforms = [A.Compose([transform]) for transform in selected_transforms]
    
    
    for transform in selected_transforms:
        Preprocess(transform, ratio=ratio).process_image()
    
    # 单独处理 MixUp
    mixup_transforms = [
        'mixup_0.2',
        'mixup_0.4'
        # 根据需要添加或移除 MixUp 策略
    ]
    
    for mixup_name in mixup_transforms:
        if mixup_name in selected_transforms:
            alpha = float(mixup_name.split('_')[1])  # 提取 MixUp 的 alpha 值
            MixUp(alpha, ratio=ratio).process_image()