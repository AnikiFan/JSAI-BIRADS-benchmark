import os
import cv2
import shutil
import random
import math
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
        print(f"警告: 无法读取图像 {image_path}")
        return None, None

    # 读取标签
    with open(label_path, "r", encoding="utf-8") as f:
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
    with open(label_save_path, "w", encoding="utf-8") as f:
        f.write(label)


def apply_augmentations(
    image, label, augmentations, image_base_name, images_dst_path, labels_dst_path, i
):
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
        aug_image = augmented["image"]

        # 构造增广后的文件名
        aug_image_file = f"{image_base_name}_aug_{i}_{idx}.jpg"
        aug_label_file = f"{image_base_name}_aug_{i}_{idx}.txt"

        # 保存增广后的图像和标签
        aug_image_path = os.path.join(images_dst_path, aug_image_file)
        aug_label_path = os.path.join(labels_dst_path, aug_label_file)
        save_image_and_label(aug_image, label, aug_image_path, aug_label_path)

        # 输出增强信息（可选）
        # print(f"已增强图像: {aug_image_file} 使用增广策略: {aug.__class__.__name__}")


def perform_mixup(
    image,
    label,
    class_names,
    class_to_images,
    src_root,
    images_dst_path,
    labels_dst_path,
    image_base_name,
    i,
    mixup_alpha,
):
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
    mix_image_src_file = os.path.join(
        src_root, mix_class_name, "images", mix_image_file
    )
    mix_label_src_file = os.path.join(
        src_root, mix_class_name, "labels", os.path.splitext(mix_image_file)[0] + ".txt"
    )

    # 读取 Mixup 图像和标签
    mix_image, mix_label = read_image_and_label(mix_image_src_file, mix_label_src_file)
    if mix_image is None:
        print(f"跳过 Mixup，因为无法读取图像: {mix_image_src_file}")
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
    with open(mixup_label_path, "w", encoding="utf-8") as f:
        f.write(f"{label} {lam}\n{mix_label} {1 - lam}")

    # 输出 Mixup 信息（可选）
    # print(f"已进行 Mixup 增强: {mixup_image_file} 混合类别: {label} 和 {mix_label} 权重: {lam:.2f}/{1 - lam:.2f}")


def compute_target_counts(class_proportions, max_images_per_class):
    """
    根据类别比例和最大图像数计算每个类别的目标图像数量。

    Args:
        class_proportions (dict): 每个类别的比例。
        max_images_per_class (dict): 每个类别的最大图像数。

    Returns:
        target_counts (dict): 每个类别的目标图像数量。
    """
    # 计算可能的总图像数量，确保不超过任何类别的最大限制
    n_total_list = [
        max_images_per_class[class_name] / p
        for class_name, p in class_proportions.items()
    ]
    N_total = min(n_total_list)

    # 根据比例计算每个类别的目标图像数量
    target_counts = {
        class_name: int(p * N_total) for class_name, p in class_proportions.items()
    }

    # 确保目标数量不超过每个类别的最大限制
    for class_name in target_counts:
        if target_counts[class_name] > max_images_per_class[class_name]:
            target_counts[class_name] = max_images_per_class[class_name]

    return target_counts


def calculate_class_aug_times(class_to_images, target_counts):
    """
    计算每个类别需要的增强次数，以达到目标图像数量。

    Args:
        class_to_images (dict): 类别到图像文件列表的映射。
        target_counts (dict): 每个类别的目标图像数量。

    Returns:
        class_aug_times (dict): 每个类别需要的增强次数。
    """
    class_aug_times = {}
    for class_name, target_count in target_counts.items():
        original_count = len(class_to_images[class_name])
        if target_count > original_count:
            # 计算需要增加的图像数量
            needed_aug_images = target_count - original_count
            # 计算每张图片需要增强的次数，向上取整
            aug_times = math.ceil(needed_aug_images / original_count)
            class_aug_times[class_name] = aug_times
        else:
            class_aug_times[class_name] = 0  # 不需要增强
    return class_aug_times


def process_image(
    image_file,
    class_name,
    src_root,
    dst_root,
    augmentations,
    class_aug_times,
    class_names,
    class_to_images,
    use_mixup,
    mixup_alpha,
):
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
    images_src_path = os.path.join(class_src_path, "images")
    labels_src_path = os.path.join(class_src_path, "labels")

    # 定义目标图像和标签路径
    images_dst_path = os.path.join(class_dst_path, "images")
    labels_dst_path = os.path.join(class_dst_path, "labels")

    # 确保目标文件夹存在
    os.makedirs(images_dst_path, exist_ok=True)
    os.makedirs(labels_dst_path, exist_ok=True)

    image_src_file = os.path.join(images_src_path, image_file)
    label_src_file = os.path.join(
        labels_src_path, os.path.splitext(image_file)[0] + ".txt"
    )

    # 读取图像和标签
    image, label = read_image_and_label(image_src_file, label_src_file)
    if image is None:
        return  # 跳过无法读取的图像

    # 输出当前处理的图像（可选）
    # print(f"正在处理图像: {image_file} 类别: {class_name}")

    # 定义统一的尺寸
    TARGET_HEIGHT = 224
    TARGET_WIDTH = 224

    # 调整图像尺寸
    image = cv2.resize(image, (TARGET_WIDTH, TARGET_HEIGHT))

    # 保存原始图像和标签到目标文件夹
    aug_image_path = os.path.join(images_dst_path, image_file)
    aug_label_path = os.path.join(
        labels_dst_path, os.path.splitext(image_file)[0] + ".txt"
    )
    save_image_and_label(image, label, aug_image_path, aug_label_path)

    # 输出原始图像保存信息（可选）
    # print(f"已保存原始图像: {image_file} 和标签: {os.path.splitext(image_file)[0] + '.txt'}")

    # 获取图像文件的基础名称
    image_base_name = os.path.splitext(image_file)[0]

    # 获取当前类别的增强次数
    aug_times = class_aug_times.get(class_name, 0)

    # 对图像进行多次增强
    for i in range(aug_times):
        # 应用常规增广
        apply_augmentations(
            image,
            label,
            augmentations,
            image_base_name,
            images_dst_path,
            labels_dst_path,
            i,
        )

        # 如果启用 Mixup，则进行 Mixup 增广
        if use_mixup:
            perform_mixup(
                image,
                label,
                class_names,
                class_to_images,
                src_root,
                images_dst_path,
                labels_dst_path,
                image_base_name,
                i,
                mixup_alpha,
            )


def create_augmented_dataset(
    src_root,
    dst_root,
    class_proportions,
    max_images_per_class,
    augmentations,
    use_mixup=False,
    mixup_alpha=0.4,
):
    """
    创建增广后的数据集。

    Args:
        src_root (str): 源数据集的根目录。
        dst_root (str): 增广后数据集的根目录。
        class_proportions (dict): 每个类别的比例。
        max_images_per_class (dict): 每个类别的最大图像数。
        augmentations (list): 增广策略列表。
        use_mixup (bool): 是否启用 Mixup 增广。
        mixup_alpha (float): Mixup 中 Beta 分布的参数。
    """
    # 创建目标根目录
    if not os.path.exists(dst_root):
        os.makedirs(dst_root)

    # 获取所有类别名称
    class_names = [
        d for d in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, d))
    ]

    # 构建类别到图像列表的映射，方便 Mixup 使用
    class_to_images = {}

    for class_name in class_names:
        images_src_path = os.path.join(src_root, class_name, "images")
        if not os.path.exists(images_src_path):
            print(f"警告: 类别 {class_name} 下的 images 文件夹不存在。")
            class_to_images[class_name] = []
            continue
        image_files = [
            f
            for f in os.listdir(images_src_path)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]
        class_to_images[class_name] = image_files

    # 统计各类别的图片数量
    class_counts = {
        class_name: len(images) for class_name, images in class_to_images.items()
    }
    print("原始各类别图片数量：")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count}")

    # 计算每个类别的目标图像数量
    target_counts = compute_target_counts(class_proportions, max_images_per_class)
    print("\n各类别的目标图像数量（基于比例和最大限制）：")
    for class_name, count in target_counts.items():
        print(f"  {class_name}: {count}")

    # 计算每个类别需要的增强次数
    class_aug_times = calculate_class_aug_times(class_to_images, target_counts)
    print("\n各类别需要的增强次数：")
    for class_name, aug_times in class_aug_times.items():
        print(f"  {class_name}: {aug_times} 次")

    # 遍历每个类别文件夹
    for class_name in class_names:
        images_src_path = os.path.join(src_root, class_name, "images")
        image_files = class_to_images[class_name]

        for image_file in tqdm(image_files, desc=f"Processing {class_name}"):
            process_image(
                image_file,
                class_name,
                src_root,
                dst_root,
                augmentations,
                class_aug_times,
                class_names,
                class_to_images,
                use_mixup,
                mixup_alpha,
            )

    print("\n数据增广完成！增广后的数据集保存在：", dst_root)


def main():
    # 源数据集路径（原始数据集）
    src_root = "/Users/huiyangzheng/Desktop/Project/Competition/GCAIAEC2024/AIC/TDS-Net/data/乳腺分类训练数据集/train_split_0.9"

    # 目标数据集路径（增广后的数据集）
    dst_root = "/Users/huiyangzheng/Desktop/Project/Competition/GCAIAEC2024/AIC/TDS-Net/data/乳腺分类训练数据集/train_split_0.9_augmented2"

    # 原数据集各类别数量（用于参考）
    # 2类: 463
    # 3类: 878
    # 4A类: 448
    # 4B类: 295
    # 4C类: 251
    # 5类: 138

    # 定义各类别的比例（比例和应为1.0）
    class_proportions = {
        "2类": 1,  # 10%
        "3类": 1,  # 30%
        "4A类": 1,  # 20%
        "4B类": 1,  # 20%
        "4C类": 1,  # 10%
        "5类": 1,  # 10%
    }

    # 归一化
    total_proportion = sum(class_proportions.values())
    normalized_proportions = {
        class_name: count / total_proportion
        for class_name, count in class_proportions.items()
    }
    print("\n归一化后的各类别比例：")
    for class_name, proportion in normalized_proportions.items():
        print(f"  {class_name}: {proportion:.4f}")

    # 定义每个类别的最大图像数量
    max = 8000
    max_images_per_class = {
        "2类": max,
        "3类": max,
        "4A类": max,
        "4B类": max,
        "4C类": max,
        "5类": max,
    }

    # # 定义增广策略列表，包含尺寸调整和多种增广方法
    # augmentations = [
    #     # 尺寸调整
    #     A.Compose([
    #         A.Resize(height=224, width=224, p=1.0),
    #     ]),

    #     # 水平翻转
    #     A.Compose([
    #         A.HorizontalFlip(p=1.0),
    #     ]),

    #     # 随机旋转一定角度
    #     A.Compose([
    #         A.Rotate(limit=10, p=1.0),
    #     ]),

    #     # 随机亮度和对比度调整
    #     A.Compose([
    #         A.RandomBrightnessContrast(p=1.0),
    #     ]),

    #     # 透视变换
    #     A.Compose([
    #         A.Perspective(scale=(0.05, 0.1), p=1.0),
    #     ]),

    #     # 弹性变换
    #     A.Compose([
    #         A.ElasticTransform(alpha=1.0, sigma=50.0, alpha_affine=None, p=1.0),
    #     ]),
    # ]

    # 定义增广策略列表，包含多种几何变换、强度变换、弹性变换和噪声添加
    augmentations = [
        # 1. 尺寸调整
        A.Compose(
            [
                A.Resize(height=224, width=224, p=1.0),
            ],
            p=1.0,
        ),
        # 原因: 统一图像尺寸，确保模型输入的一致性。超声影像的尺寸可能不统一，通过调整大小可以避免模型处理不同尺寸的图像带来的复杂性。
        # 2. 水平和垂直翻转
        A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
            ],
            p=1.0,
        ),
        # 原因: 增加图像的多样性，模拟不同的扫描方向和患者姿势变化，提高模型对位置和方向变化的适应能力。
        # 3. 随机旋转
        A.Compose(
            [
                A.Rotate(limit=15, p=0.7),  # 随机旋转±15度
            ],
            p=1.0,
        ),
        # 原因: 模拟不同的扫描角度，增强模型对旋转变换的鲁棒性，使模型能够识别不同角度下的相同病灶。
        # 4. 随机缩放和平移
        A.Compose(
            [
                A.ShiftScaleRotate(
                    shift_limit=0.1,  # 随机平移范围为图像尺寸的10%
                    scale_limit=0.1,  # 随机缩放范围为±10%
                    rotate_limit=15,  # 随机旋转范围为±15度
                    p=0.7,
                ),
            ],
            p=1.0,
        ),
        # 原因: 通过随机缩放和平移，模拟不同的放大倍数和图像位置变化，进一步增强模型对图像变换的适应能力。
        # 5. 随机亮度和对比度调整
        A.Compose(
            [
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,  # 亮度调整范围
                    contrast_limit=0.2,  # 对比度调整范围
                    p=0.7,
                ),
            ],
            p=1.0,
        ),
        # 原因: 模拟不同的成像条件和设备设置，增强模型对亮度和对比度变化的适应能力，提高在不同成像环境下的性能。
        # 6. 伽马校正
        A.Compose(
            [
                A.GammaTransform(gamma_limit=(80, 120), p=0.5),  # 伽马值调整范围
            ],
            p=1.0,
        ),
        # 原因: 改变图像的伽马值以增强图像细节和对比度，帮助模型更好地识别细微的组织结构变化。
        # 7. 弹性变换
        A.Compose(
            [
                A.ElasticTransform(
                    alpha=1.0,  # 控制弹性变换的强度
                    sigma=50.0,  # 控制弹性变换的平滑度
                    alpha_affine=50,  # 控制仿射变换的强度
                    p=0.5,
                ),
            ],
            p=1.0,
        ),
        # 原因: 模拟组织的自然变形，增强模型对形态变化的鲁棒性，特别适用于医学影像中组织结构的多样性。
        # 8. 高斯噪声添加
        A.Compose(
            [
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),  # 高斯噪声的方差范围
            ],
            p=1.0,
        ),
        # 原因: 模拟超声成像过程中常见的噪声，提高模型对噪声的耐受性，增强模型在实际应用中的鲁棒性。
        # 9. 盐和胡椒噪声添加
        A.Compose(
            [
                A.IAAAdditiveGaussianNoise(p=0.3),
                A.IAASaltAndPepper(p=0.3),
            ],
            p=1.0,
        ),
        # 原因: 模拟不同类型的噪声，进一步增强模型的鲁棒性，使其能够处理各种噪声干扰下的图像。
        # 10. 透视变换
        A.Compose(
            [
                A.Perspective(scale=(0.05, 0.1), p=0.5),  # 透视变换的范围
            ],
            p=1.0,
        ),
        # 原因: 模拟不同的观察角度，增强模型对透视变化的适应能力，使模型能够识别不同角度下的病灶。
        # 11. Cutout（随机遮挡部分图像）
        A.Compose(
            [
                A.Cutout(
                    num_holes=8,  # 遮挡孔的数量
                    max_h_size=32,  # 遮挡孔的最大高度
                    max_w_size=32,  # 遮挡孔的最大宽度
                    fill_value=0,  # 遮挡部分的填充值
                    p=0.5,
                ),
            ],
            p=1.0,
        ),
        # 原因: 随机遮挡图像的部分区域，迫使模型关注更多的特征区域，提高模型的泛化能力，防止过拟合。
        # 12. 随机饱和度调整（适用于彩色图像）
        A.Compose(
            [
                A.HueSaturationValue(
                    hue_shift_limit=20,  # 色调调整范围
                    sat_shift_limit=30,  # 饱和度调整范围
                    val_shift_limit=20,  # 明度调整范围
                    p=0.5,
                ),
            ],
            p=1.0,
        ),
        # 原因: 增加颜色的多样性，模拟不同的成像条件（适用于彩色超声图像），帮助模型更好地适应颜色变化。
        # 13. 随机裁剪和缩放
        A.Compose(
            [
                A.RandomResizedCrop(
                    height=224,
                    width=224,
                    scale=(0.8, 1.0),  # 随机裁剪的缩放比例
                    ratio=(0.9, 1.1),  # 随机裁剪的长宽比
                    p=0.5,
                ),
            ],
            p=1.0,
        ),
        # 原因: 随机裁剪图像并重新缩放，进一步增加图像的多样性，帮助模型学习到不同区域和比例下的特征。
    ]
    # 启用 Mixup，并设置 alpha 参数
    use_mixup = True
    mixup_alpha = 0.3  # Beta 分布的参数，控制混合比例

    create_augmented_dataset(
        src_root,
        dst_root,
        class_proportions,
        max_images_per_class,
        augmentations,
        use_mixup,
        mixup_alpha,
    )


if __name__ == "__main__":
    main()
