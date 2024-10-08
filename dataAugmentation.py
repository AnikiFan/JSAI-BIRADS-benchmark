from utils.dataAugmentation import Preprocess
from utils.dataAugmentation import MixUp
import albumentations as A



if __name__ == '__main__':
    ratio = [2, 1, 8, 7, 7, 6]

    # 1. 旋转 (Rotation)
    transform_rotate = A.Compose([
        A.Rotate(limit=15, p=1.0)  # 始终应用旋转
    ])
    Preprocess(transform_rotate, ratio=ratio).process_image()

    # 2. 水平翻转 (Horizontal Flip)
    transform_hflip = A.Compose([
        A.HorizontalFlip(p=1.0)  # 始终应用水平翻转
    ])
    Preprocess(transform_hflip, ratio=ratio).process_image()

    # 3. 垂直翻转 (Vertical Flip)
    transform_vflip = A.Compose([
        A.VerticalFlip(p=1.0)  # 始终应用垂直翻转
    ])
    Preprocess(transform_vflip, ratio=ratio).process_image()

    # 4. 随机亮度和对比度 (Random Brightness and Contrast)
    transform_brightness_contrast = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0)
    ])
    Preprocess(transform_brightness_contrast, ratio=ratio).process_image()

    # 5. 高斯噪声 (Gaussian Noise)
    transform_gaussian_noise = A.Compose([
        A.GaussNoise(var_limit=(10.0, 50.0), p=1.0)
    ])
    Preprocess(transform_gaussian_noise, ratio=ratio).process_image()

    # 6. 弹性变换 (Elastic Transform)
    transform_elastic = A.Compose([
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=None, p=1.0)
    ])
    Preprocess(transform_elastic, ratio=ratio).process_image()

    # 7. CLAHE
    transform_clahe = A.Compose([
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0)
    ])
    Preprocess(transform_clahe, ratio=ratio).process_image()

    # 8. Cutout
    # transform_cutout = A.Compose([
    #     A.cutout(num_holes=8, max_h_size=16, max_w_size=16, p=1.0)
    # ])
    # Preprocess(transform_cutout, ratio=ratio).process_image()

    # 9. 模糊 (Gaussian Blur)
    transform_blur = A.Compose([
        A.GaussianBlur(blur_limit=(3, 7), p=1.0)
    ])
    Preprocess(transform_blur, ratio=ratio).process_image()

    # 10. 随机缩放和平移 (Random Scale and Translate)
    transform_shift_scale_rotate = A.Compose([
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=1.0)
    ])
    Preprocess(transform_shift_scale_rotate, ratio=ratio).process_image()

    # 11. MixUp（如果需要，可以单独处理或集成到训练循环中）
    # MixUp通常在数据加载时动态应用，不需要预处理
    # 若需要预处理，可以在此处调用相应的方法
    # from utils.dataAugmentation import MixUp
    MixUp(0.2, ratio=ratio).process_image()

    # 12. 随机擦除 (Random Erasing)
    # transform_random_erasing = A.Compose([
    #     A.RandomErasing(p=1.0)
    # ])
    # Preprocess(transform_random_erasing, ratio=ratio).process_image()