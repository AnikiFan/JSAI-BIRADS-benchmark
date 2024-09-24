import os
import cv2
import shutil
import random
import numpy as np
import albumentations as A
from tqdm import tqdm

def read_image_and_label(image_path, label_path):
    """
    读取图像和标签。

    Args:
        image_path (str): 图像文件的路径。
        label_path (str): 标签文件的路径。

    Returns:
        image (numpy.ndarray): 读取的图像。
        label (str): 图像的标签。
    """
    # 读取图像
    image = cv2.imread(image_path)

    # 检查图像是否成功读取
    if image is None:
        print(f"Warning: Failed to read image {image_path}")
        return None, None

    # 读取标签
    with open(label_path, 'r', encoding='utf-8') as f:
        label = f.read().strip()

    return image, label

def save_image_and_label(image, label, image_save_path, label_save_path):
    """
    保存图像和标签。

    Args:
        image (numpy.ndarray): 要保存的图像。
        label (str): 要保存的标签。
        image_save_path (str): 图像保存路径。
        label_save_path (str): 标签保存路径。
    """
    # 保存图像
    cv2.imwrite(image_save_path, image)

    # 保存标签
    with open(label_save_path, 'w', encoding='utf-8') as f:
        f.write(label)

def apply_augmentations(image, label, augmentations, image_base_name, images_dst_path, labels_dst_path, i):
    """
    对图像应用增广并保存。

    Args:
        image (numpy.ndarray): 原始图像。
        label (str): 原始标签。
        augmentations (list): 增广策略列表。
        image_base_name (str): 图像文件的基础名称（不包含扩展名）。
        images_dst_path (str): 增广后图像的保存路径。
        labels_dst_path (str): 增广后标签的保存路径。
        i (int): 增广次数的索引。
    """
    for idx, aug in enumerate(augmentations):
        # 应用增广
        augmented = aug(image=image)
        aug_image = augmented['image']

        # 构造增广后的文件名
        aug_image_file = f"{image_base_name}_aug_{i}_{idx}.jpg"
        aug_label_file = f"{image_base_name}_aug_{i}_{idx}.txt"

        # 保存增广后的图像和标签
        aug_image_path = os.path.join(images_dst_path, aug_image_file)
        aug_label_path = os.path.join(labels_dst_path, aug_label_file)
        save_image_and_label(aug_image, label, aug_image_path, aug_label_path)

def perform_mixup(image, label, class_names, class_to_images, src_root, images_dst_path, labels_dst_path, image_base_name, i, mixup_alpha):
    """
    对图像进行 Mixup 增广并保存。

    Args:
        image (numpy.ndarray): 原始图像。
        label (str): 原始标签。
        class_names (list): 所有类别的名称列表。
        class_to_images (dict): 类别到图像文件列表的映射。
        src_root (str): 源数据集的根目录。
        images_dst_path (str): 增广后图像的保存路径。
        labels_dst_path (str): 增广后标签的保存路径。
        image_base_name (str): 图像文件的基础名称（不包含扩展名）。
        i (int): 增广次数的索引。
        mixup_alpha (float): Mixup 中 Beta 分布的参数。
    """
    # 随机选择另一张图像进行 Mixup
    mix_class_name = random.choice(class_names)
    mix_image_file = random.choice(class_to_images[mix_class_name])
    mix_image_src_file = os.path.join(src_root, mix_class_name, 'images', mix_image_file)
    mix_label_src_file = os.path.join(src_root, mix_class_name, 'labels', os.path.splitext(mix_image_file)[0] + '.txt')

    # 读取 Mixup 图像和标签
    mix_image, mix_label = read_image_and_label(mix_image_src_file, mix_label_src_file)
    if mix_image is None:
        return  # 跳过无法读取的图像

    # 调整 Mixup 图像尺寸为原图像尺寸
    mix_image = cv2.resize(mix_image, (image.shape[1], image.shape[0]))

    # 计算 Mixup 权重
    lam = np.random.beta(mixup_alpha, mixup_alpha)

    # 进行 Mixup
    mixup_image = (lam * image + (1 - lam) * mix_image).astype(np.uint8)

    # 构造 Mixup 后的文件名
    mixup_image_file = f"{image_base_name}_mixup_{i}.jpg"
    mixup_label_file = f"{image_base_name}_mixup_{i}.txt"

    # 保存 Mixup 后的图像
    mixup_image_path = os.path.join(images_dst_path, mixup_image_file)
    cv2.imwrite(mixup_image_path, mixup_image)

    # 混合标签（对于分类任务，可以保存两个标签和权重）
    mixup_label_path = os.path.join(labels_dst_path, mixup_label_file)
    with open(mixup_label_path, 'w', encoding='utf-8') as f:
        f.write(f"{label} {lam}\n{mix_label} {1 - lam}")

def process_image(image_file, class_name, src_root, dst_root, augmentations, class_aug_times, class_names, class_to_images, use_mixup, mixup_alpha):
    """
    处理单张图像，包括读取、增广、Mixup 和保存。

    Args:
        image_file (str): 图像文件名。
        class_name (str): 图像所属的类别名称。
        src_root (str): 源数据集的根目录。
        dst_root (str): 增广后数据集的根目录。
        augmentations (list): 增广策略列表。
        class_aug_times (dict): 每个类别的增广次数。
        class_names (list): 所有类别的名称列表。
        class_to_images (dict): 类别到图像文件列表的映射。
        use_mixup (bool): 是否启用 Mixup 增广。
        mixup_alpha (float): Mixup 中 Beta 分布的参数。
    """
    class_src_path = os.path.join(src_root, class_name)
    class_dst_path = os.path.join(dst_root, class_name)

    # 定义源图像和标签路径
    images_src_path = os.path.join(class_src_path, 'images')
    labels_src_path = os.path.join(class_src_path, 'labels')

    # 定义目标图像和标签路径
    images_dst_path = os.path.join(class_dst_path, 'images')
    labels_dst_path = os.path.join(class_dst_path, 'labels')

    # 确保目标文件夹存在
    os.makedirs(images_dst_path, exist_ok=True)
    os.makedirs(labels_dst_path, exist_ok=True)

    image_src_file = os.path.join(images_src_path, image_file)
    label_src_file = os.path.join(labels_src_path, os.path.splitext(image_file)[0] + '.txt')

    # 读取图像和标签
    image, label = read_image_and_label(image_src_file, label_src_file)
    if image is None:
        return  # 跳过无法读取的图像

    # 定义统一的尺寸
    TARGET_HEIGHT = 224
    TARGET_WIDTH = 224

    # 调整图像尺寸
    image = cv2.resize(image, (TARGET_WIDTH, TARGET_HEIGHT))

    # 保存原始图像和标签到目标文件夹
    aug_image_path = os.path.join(images_dst_path, image_file)
    aug_label_path = os.path.join(labels_dst_path, os.path.splitext(image_file)[0] + '.txt')
    save_image_and_label(image, label, aug_image_path, aug_label_path)

    # 获取图像文件的基础名称
    image_base_name = os.path.splitext(image_file)[0]

    # 获取当前类别的增广次数
    aug_times = class_aug_times.get(class_name, 1)

    # 对图像进行多次增广
    for i in range(aug_times):
        # 应用常规增广
        apply_augmentations(image, label, augmentations, image_base_name, images_dst_path, labels_dst_path, i)

        # 如果启用 Mixup，则进行 Mixup 增广
        if use_mixup:
            perform_mixup(image, label, class_names, class_to_images, src_root, images_dst_path, labels_dst_path, image_base_name, i, mixup_alpha)

def create_augmented_dataset(src_root, dst_root, class_aug_times, augmentations, use_mixup=False, mixup_alpha=0.4):
    """
    创建增广后的数据集。

    Args:
        src_root (str): 源数据集的根目录。
        dst_root (str): 增广后数据集的根目录。
        class_aug_times (dict): 每个类别的增广次数。
        augmentations (list): 增广策略列表。
        use_mixup (bool): 是否启用 Mixup 增广。
        mixup_alpha (float): Mixup 中 Beta 分布的参数。
    """
    
    # 创建目标根目录
    if not os.path.exists(dst_root):
        os.makedirs(dst_root)

    # 获取所有类别名称
    class_names = [d for d in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, d))]

    # 构建类别到图像列表的映射，方便 Mixup 使用
    class_to_images = {}

    for class_name in class_names:
        images_src_path = os.path.join(src_root, class_name, 'images')
        image_files = [f for f in os.listdir(images_src_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        class_to_images[class_name] = image_files

    # 遍历每个类别文件夹
    for class_name in class_names:
        images_src_path = os.path.join(src_root, class_name, 'images')
        image_files = class_to_images[class_name]

        for image_file in tqdm(image_files, desc=f'Processing {class_name}'):
            process_image(
                image_file, class_name, src_root, dst_root,
                augmentations, class_aug_times, class_names,
                class_to_images, use_mixup, mixup_alpha
            )

    print('数据增广完成！增广后的数据集保存在：', dst_root)

def main():
    # 源数据集路径（原始数据集）
    src_root = '/Users/huiyangzheng/Desktop/Project/Competition/GCAIAEC2024/Data/乳腺分类训练数据集/train'

    # 目标数据集路径（增广后的数据集）
    dst_root = '/Users/huiyangzheng/Desktop/Project/Competition/GCAIAEC2024/Data/乳腺分类训练数据集/train_augmented'

    # 原数据集各类别数量：
    # 2类: 463
    # 3类: 878
    # 4A类: 448
    # 4B类: 295
    # 4C类: 251
    # 5类: 138
    # 定义每个类别的增广次数，针对少数类别进行更多增广
    class_aug_times = {
        '2类': 2,   # 463
        '3类': 1,   # 878
        '4A类': 3,  # 448
        '4B类': 4,  # 295
        '4C类': 5,  # 251
        '5类': 6    # 138
    }

    # 定义增广策略列表，包含尺寸调整和多种增广方法
    augmentations = [
        # 尺寸调整
        A.Compose([
            A.Resize(height=224, width=224, p=1.0),
        ]),

        # 水平翻转
        A.Compose([
            A.HorizontalFlip(p=1.0),
        ]),

        # 垂直翻转
        # A.Compose([
        #     A.VerticalFlip(p=1.0),
        # ]),

        # 随机旋转一定角度
        A.Compose([
            A.Rotate(limit=10, p=1.0),
        ]),

        # 随机亮度和对比度调整
        A.Compose([
            A.RandomBrightnessContrast(p=1.0),
        ]),

        # # 高斯噪声
        # A.Compose([
        #     A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
        # ]),

        # 仿射变换
        # A.Compose([
        #     A.Affine(scale=(0.9, 1.1), translate_percent=(0.1, 0.1), rotate=(-15, 15), shear=(-10, 10), p=1.0),
        # ]),

        # 颜色抖动，乳腺超声影像为灰度图，颜色抖动对图像无实际意义，建议移除。
        # A.Compose([
        #     A.ColorJitter(p=1.0),
        # ]),

        # 随机擦除
        # A.Compose([
        #     A.CoarseDropout(max_holes=8, max_height=16, max_width=16, fill_value=0, p=1.0),
        # ]),

        # 透视变换
        A.Compose([
            A.Perspective(scale=(0.05, 0.1), p=1.0),
        ]),

        # 弹性变换
        A.Compose([
            A.ElasticTransform(alpha=1.0, sigma=50.0, alpha_affine=None, p=1.0),
        ]),
    ]

    # 启用 Mixup，并设置 alpha 参数
    use_mixup = True
    mixup_alpha = 0.4  # Beta 分布的参数，控制混合比例

    create_augmented_dataset(src_root, dst_root, class_aug_times, augmentations, use_mixup, mixup_alpha)

if __name__ == '__main__':
    main()