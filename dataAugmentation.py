from utils.dataAugmentation import Preprocess, MixUp
import albumentations as A
import numpy as np
# from debug import debug

'''
原始数据集train(划分后)
1    1061
0     849
2     404
3     274
5     269
4     232

原始数据集（未划分）
1    1349
0    1035
2     492
5     343
3     341
4     301
'''


    

if __name__ == '__main__':
    # initial_num = np.array([849, 1061, 404, 274, 232, 269])
    initial_num = np.array([1035, 1349, 492, 341, 301, 343])
    target_num = np.array([200, 200, 200, 200, 200, 200])
    print("target_num: ", target_num)
    ratio = (target_num) / initial_num
    ratio = ratio.tolist()
    
    print("ratio: ", ratio)
    
    # 设置要使用的增广策略
    selected_transforms = [
        # A.Rotate(limit=10, p=1.0),
        # A.HorizontalFlip(p=1.0),
        # A.VerticalFlip(p=1.0),
        # A.RandomBrightnessContrast(brightness_limit=
        # 0.2, contrast_limit=0.2, p=1.0),
        # A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
        # A.ElasticTransform(alpha=1.0, sigma=50, p=1.0),
        # A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        # A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=1.0)
        # A.Rotate(limit=10, p=1.0),
        # A.HorizontalFlip(p=1.0), 
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
        A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
        A.ElasticTransform(alpha=1.0, sigma=50, alpha_affine=50, p=1.0),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=0, p=1.0),
        A.RandomGamma(gamma_limit=(80, 120), p=1.0)
        # A.Compose([
        #     A.Rotate(limit=10, p=0.5),
        #     A.HorizontalFlip(p=0.5), 
        #     A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        #     A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        #     A.ElasticTransform(alpha=1.0, sigma=50, alpha_affine=50, p=0.5),
        #     A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=0, p=0.5),
        #     A.RandomGamma(gamma_limit=(80, 120), p=0.5)
        # ])
    ]
    
    # 转化成compose
    selected_transforms = [A.Compose([transform]) for transform in selected_transforms]
    
    
    for transform in selected_transforms:
        Preprocess(transform, ratio=ratio).process_image()
    
    # allData_num = 
    mixup_target_num = np.array([200, 200, 200, 200, 200, 200])
    mixup_ratio = mixup_target_num / initial_num
    mixup_ratio = mixup_ratio.tolist()
    
    # 单独处理 MixUp
    mixup_transforms = [
        'mixup_0.2',
        # 'mixup_0.3'
        # 根据需要添加或移除 MixUp 策略
    ]
    
    for mixup_name in mixup_transforms:
        alpha = float(mixup_name.split('_')[1])  # 提取 MixUp 的 alpha 值
        MixUp(alpha, ratio=mixup_ratio).process_image()
    
    