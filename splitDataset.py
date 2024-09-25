import os
import shutil
import random

def split_dataset(dataset_dir, train_dir, test_dir, split_ratio=0.8, seed=42):
    """
    将数据集按指定比例划分为训练集和测试集。

    参数：
    - dataset_dir: 原始数据集目录（包含各类别子目录）
    - train_dir: 训练集输出目录
    - test_dir: 测试集输出目录
    - split_ratio: 训练集所占比例（默认80%）
    - seed: 随机种子（保证可重复性，默认42）
    """
    # 设置随机种子
    random.seed(seed)
    
    # 创建训练集和测试集的根目录
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # 遍历每个类别
    for class_name in os.listdir(dataset_dir):
        class_path = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_path):
            continue  # 跳过非目录文件
        
        images_path = os.path.join(class_path, 'images')
        labels_path = os.path.join(class_path, 'labels')
        
        # 检查images和labels目录是否存在
        if not os.path.exists(images_path) or not os.path.exists(labels_path):
            print(f"跳过 {class_name}，缺少 images 或 labels 目录。")
            continue
        
        # 列出所有图像文件（支持.jpg和.png）
        image_files = [f for f in os.listdir(images_path) 
                      if os.path.isfile(os.path.join(images_path, f)) and f.lower().endswith(('.jpg', '.png'))]
        
        # 如果没有图像文件，跳过
        if not image_files:
            print(f"类别 {class_name} 中没有图像文件，跳过。")
            continue
        
        # 随机打乱图像文件列表
        random.shuffle(image_files)
        
        # 计算分割索引
        split_index = int(len(image_files) * split_ratio)
        train_files = image_files[:split_index]
        test_files = image_files[split_index:]
        
        # 定义目标目录
        train_class_images = os.path.join(train_dir, class_name, 'images')
        train_class_labels = os.path.join(train_dir, class_name, 'labels')
        test_class_images = os.path.join(test_dir, class_name, 'images')
        test_class_labels = os.path.join(test_dir, class_name, 'labels')
        
        # 创建目标目录
        os.makedirs(train_class_images, exist_ok=True)
        os.makedirs(train_class_labels, exist_ok=True)
        os.makedirs(test_class_images, exist_ok=True)
        os.makedirs(test_class_labels, exist_ok=True)
        
        # 复制训练集文件
        for img in train_files:
            src_img = os.path.join(images_path, img)
            dst_img = os.path.join(train_class_images, img)
            shutil.copy2(src_img, dst_img)
            
            # 复制对应的标签文件
            base, _ = os.path.splitext(img)
            label_file = base + '.txt'
            src_label = os.path.join(labels_path, label_file)
            if os.path.exists(src_label):
                dst_label = os.path.join(train_class_labels, label_file)
                shutil.copy2(src_label, dst_label)
            else:
                print(f"警告：标签文件 {src_label} 不存在。")
        
        # 复制测试集文件
        for img in test_files:
            src_img = os.path.join(images_path, img)
            dst_img = os.path.join(test_class_images, img)
            shutil.copy2(src_img, dst_img)
            
            # 复制对应的标签文件
            base, _ = os.path.splitext(img)
            label_file = base + '.txt'
            src_label = os.path.join(labels_path, label_file)
            if os.path.exists(src_label):
                dst_label = os.path.join(test_class_labels, label_file)
                shutil.copy2(src_label, dst_label)
            else:
                print(f"警告：标签文件 {src_label} 不存在。")
                
        print(f"类别 {class_name}：{len(train_files)} 个训练样本，{len(test_files)} 个测试样本。")

# 示例用法
if __name__ == "__main__":
    # 定义原始数据集目录
    original_dataset_dir = '/Users/huiyangzheng/Desktop/Project/Competition/GCAIAEC2024/AIC/TDS-Net/data/乳腺分类训练数据集/train'  # 根据实际情况修改
    
    split_ratio = 0.9
    
    # 定义输出训练集和测试集目录
    output_train_dir = os.path.join(os.path.dirname(original_dataset_dir), 'train_split', 'split_train')
    output_test_dir = os.path.join(os.path.dirname(original_dataset_dir), 'train_split', 'split_test')
    
    # 创建输出目录
    os.makedirs(output_train_dir, exist_ok=True)
    os.makedirs(output_test_dir, exist_ok=True)
    
    # 同时写入text来保存相关信息
    text_dir = os.path.join(os.path.dirname(original_dataset_dir), 'train_split', 'split_info.txt')
    
    # 创建text文件所在目录
    os.makedirs(os.path.dirname(text_dir), exist_ok=True)
    
    # 写入分割信息
    with open(text_dir, 'w') as f:  
        f.write(f'train_dir: {output_train_dir}\n')
        f.write(f'test_dir: {output_test_dir}\n')
        f.write(f'split_ratio: {split_ratio}\n')
    
    # 调用函数进行数据集划分
    split_dataset(original_dataset_dir, output_train_dir, output_test_dir, split_ratio=split_ratio) 