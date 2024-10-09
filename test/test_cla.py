import os
import torch
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from datetime import datetime

# 导入您项目中的自定义模块和类
from models.UnetClassifer.unet import PretrainedClassifier
from models.model4compare.AlexNet import AlexNet
from models.model4compare.GoogleNet import GoogleNet
from models.model4compare.VGG import VGG
from models.model4compare.NiN import NiN
from TDSNet.TDSNet import TDSNet

from utils.tools import create_transforms  # 确保这个函数可用
from utils.MyBlock.MyCrop import MyCrop
from utils.PILResize import PILResize

# 定义测试配置字典
test_cfg = {
    "model": "Unet",  # 模型选择，与训练时相同
    "data": "Breast",  # 数据集选择
    "csv_path": "./data/breast/test.csv",  # 测试CSV文件路径
    "images_dir": "./data/breast/train_valid_test/A",  # 测试图片根目录
    "checkpoint_path": "./checkPoint/Unet_Breast_20230427_123456/model_best.pth",  # 模型检查点路径
    "output_csv": "./checkPoint/Unet_Breast_20230427_123456/test_predictions.csv",  # 输出预测结果CSV文件路径
    "batch_size": 16,  # 测试时的批处理大小
    "num_workers": 2,  # 数据加载器的工作进程数量
    "device": "cuda" if torch.cuda.is_available() else "cpu",  # 设备选择
    "transforms_cfg": {
        "transform_test": {
            "PILResize": {"size": (400, 400)},
            "MyCrop": {}, 
            "ToTensor": {},
            "Normalize": {
                "mean": [0.4914, 0.4822, 0.4465],
                "std": [0.2023, 0.1994, 0.2010],
            },
        },
    },
    "custom_transforms": {
        'MyCrop': MyCrop,
        'PILResize': PILResize
    },
    "num_classes": 6,  # 类别数量，根据您的实际情况设置
    "class_names": ['2类', '3类', '4A类', '4B类', '4C类', '5类']  # 类别名称列表
}

class BreastTestDataset(Dataset):
    def __init__(self, csv_file, images_dir, transform=None, allowed_extensions=None):
        """
        自定义测试数据集
        Args:
            csv_file (str): CSV文件的路径。
            images_dir (str): 存放测试图片的根目录。
            transform (callable, optional): 可选的变换。
            allowed_extensions (set, optional): 允许的图片文件扩展名。
        """
        self.df = pd.read_csv(csv_file)
        self.images_dir = images_dir
        self.transform = transform
        self.allowed_extensions = allowed_extensions or {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        self.image_paths = self._map_image_names()

    def _map_image_names(self):
        """
        遍历 images_dir 并创建图片名到完整路径的映射字典，仅包含允许的图片文件类型。
        """
        image_map = {}
        for root, dirs, files in os.walk(self.images_dir):
            for file in files:
                file_lower = file.lower()
                ext = os.path.splitext(file_lower)[1]
                if ext in self.allowed_extensions:
                    image_map[file.strip()] = os.path.join(root, file)
        print(f"Total images found (filtered by extensions {self.allowed_extensions}): {len(image_map)}")
        # 打印映射字典的一部分
        print("===== 图片路径映射示例 =====")
        sample = list(image_map.items())[:10]  # 打印前10个映射
        for img, path in sample:
            print(f"{img}: {path}")
        return image_map

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.df.iloc[idx]['image_name'].strip()
        img_category = self.df.iloc[idx]['category'].strip()

        # 获取图片完整路径
        image_path = self.image_paths.get(img_name)
        if image_path is None:
            raise FileNotFoundError(f"Image {img_name} not found in {self.images_dir}")

        # 加载并转换图片
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, idx  # 返回图像和索引，以便后续写入结果

def modelSelector(model_name, checkpoint_path, num_class, device):
    """
    根据模型名称加载对应的模型并加载权重。
    Args:
        model_name (str): 模型名称，如 'Unet'。
        checkpoint_path (str): 模型检查点路径。
        num_class (int): 类别数量。
        device (str): 设备，如 'cuda' 或 'cpu'。
    Returns:
        model: 加载好的模型。
    """
    if model_name == 'TDSNet':
        model = TDSNet(num_class)
    elif model_name == 'AlexNet':
        model = AlexNet(num_class)
    elif model_name == 'GoogleNet':
        model = GoogleNet(num_class)
    elif model_name == 'VGG':
        model = VGG(((1, 64), (1, 128), (2, 256), (2, 512), (2, 512)), num_class)
    elif model_name == 'NiN':
        model = NiN(num_class)
    elif model_name == 'Unet':
        model = PretrainedClassifier(num_classes=num_class, in_channels=3,
                               backbone=test_cfg['transforms_cfg']['transform_test'].get("backbone", "resnet50"),
                               pretrained=True)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # 加载模型权重
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# def check_dataset(csv_file, images_dir):
#     """
#     检查数据集的完整性，确保所有图像文件都存在于指定目录中。
#     Args:
#         csv_file (str): CSV文件的路径。
#         images_dir (str): 存放测试图片的根目录。
#     """
#     df = pd.read_csv(csv_file)
#     missing_images = []

#     for img_name in df['image_name']:
#         image_path = os.path.join(images_dir, img_name)
#         if not os.path.isfile(image_path):
#             missing_images.append(img_name)

#     if missing_images:
#         print(f"缺失的图像文件: {missing_images}")
#     else:
#         print("所有图像文件均存在。")
def verify_breast_test_dataset(dataset, images_dir, csv_path):
    """
    验证BreastTestDataset对象，确保所有图片都能正确加载。
    
    Args:
        dataset (BreastTestDataset): 自定义测试数据集实例。
        images_dir (str): 存放测试图片的根目录。
        csv_path (str): CSV文件的路径。
    
    Returns:
        None
    """
    # 读取CSV文件
    try:
        csv_df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"无法读取CSV文件: {e}")
        return
    
    csv_images = set(csv_df['image_name'].str.strip())
    
    # 遍历images_dir，收集所有图片文件名
    actual_images = set()
    for root, dirs, files in os.walk(images_dir):
        for file in files:
            file_lower = file.lower()
            ext = os.path.splitext(file_lower)[1]
            if ext in dataset.allowed_extensions:
                actual_images.add(file.strip())
    
    # 找出缺失的图片
    missing_images = csv_images - actual_images
    # 找出多余的图片
    extra_images = actual_images - csv_images
    
    print("\n===== 数据集验证结果 =====")
    print(f"CSV文件中的图片总数: {len(csv_images)}")
    print(f"实际目录中的图片总数: {len(actual_images)}")
    print(f"缺失的图片数量: {len(missing_images)}")
    print(f"多余的图片数量: {len(extra_images)}")
    
    if missing_images:
        print("\n缺失的图片列表（显示前10个）：")
        for img in list(missing_images)[:10]:
            print(f"- {img}")
    else:
        print("\n所有CSV文件中的图片都存在于指定的images_dir中。")
    
    if extra_images:
        print("\n多余的图片列表（显示前10个）：")
        for img in list(extra_images)[:10]:
            print(f"- {img}")
    else:
        print("\nimages_dir中没有额外的图片。")
    
    # 进一步检查每个图片是否能被成功加载
    print("\n===== 尝试加载所有图片 =====")
    missing_count = 0
    loaded_count = 0
    for idx in range(len(dataset)):
        img_name = csv_df.iloc[idx]['image_name'].strip()
        image_path = dataset.image_paths.get(img_name)
        # 输出部分图片的尺寸
        if idx%100 == 0:
            print(f"Image '{img_name}' at index {idx}: {image_path}")
            print(f"Image shape: {Image.open(image_path).size}")
        if image_path and os.path.isfile(image_path):
            try:
                image, index = dataset[idx]
                loaded_count += 1
            except Exception as e:
                print(f"Error loading image '{img_name}' at index {idx}: {e}")
                missing_count += 1
        else:
            print(f"Image '{img_name}' not found at expected path.")
            missing_count += 1
    
    print(f"\n成功加载的图片数量: {loaded_count}")
    print(f"加载失败的图片数量: {missing_count}")

    if missing_count == 0:
        print("所有图片都成功加载。")
    else:
        print("存在加载失败的图片，请检查路径配置和图片文件。")

def check_breast_test_dataset(dataset):
    """
    检查BreastTestDataset的返回结果，确保所有图像都能正确加载。
    Args:
        dataset (BreastTestDataset): 自定义测试数据集实例。
    """
    for idx in range(len(dataset)):
        try:
            image, index = dataset[idx]
            print(f"Index: {index}, Image shape: {image.size}")
        except Exception as e:
            print(f"Error loading image at index {idx}: {e}")

def main():
    # 设备配置
    device = torch.device(test_cfg["device"])
    print(f"Using device: {device}")

    # 读取CSV
    df = pd.read_csv(test_cfg["csv_path"])
    print(f"Total test samples: {len(df)}")

    # 定义图像变换
    transform_test = create_transforms(test_cfg["transforms_cfg"]["transform_test"], 
                                       custom_transforms=test_cfg["custom_transforms"])

    # 创建测试数据集和数据加载器
    test_dataset = BreastTestDataset(csv_file=test_cfg["csv_path"], 
                                     images_dir=test_cfg["images_dir"], 
                                     transform=transform_test)
    
    # check_dataset(test_cfg["csv_path"], test_cfg["images_dir"])
    check_breast_test_dataset(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=test_cfg["batch_size"], 
                             shuffle=False, num_workers=test_cfg["num_workers"])

    # 加载模型
    model = modelSelector(test_cfg["model"], test_cfg["checkpoint_path"], 
                          test_cfg["num_classes"], device)
    print(f"Loaded model {test_cfg['model']} from {test_cfg['checkpoint_path']}")

    # 初始化预测列表
    predictions = []

    # 进行预测
    with torch.no_grad():
        for images, indices in tqdm(test_loader, desc="Predicting"):
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            preds = preds.cpu().numpy()
            predictions.extend(preds)

    # 将预测结果映射为类别名称
    # 确保class_names的索引与模型输出对应
    class_names = test_cfg["class_names"]
    df['predicted_category'] = [class_names[p] for p in predictions]

    # 保存到新的CSV文件
    os.makedirs(os.path.dirname(test_cfg["output_csv"]), exist_ok=True)
    df.to_csv(test_cfg["output_csv"], index=False)
    print(f"Predictions saved to {test_cfg['output_csv']}")




if __name__ == '__main__':
    main()