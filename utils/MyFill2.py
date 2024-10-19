import os.path
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch
import torchvision.transforms as T
from torchvision.io import read_image
import torchvision.transforms.functional as TF
from PIL import Image  # 加入PIL用于处理输入MyCrop是PIL Image而不是Tensor的情况
from tqdm import tqdm

debug = False


def show(imgs,image_name=None, save_folder_path=None, display=True):
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)
    ncols, nrows = (len(imgs) + 1) // 2, 2
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, squeeze=False)
    for i, img in enumerate(imgs):
        if isinstance(img, torch.Tensor):
            img = T.ToPILImage()(img.to('cpu'))
        axs[i // ncols, i % ncols].imshow(np.asarray(img))
        axs[i // ncols, i % ncols].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    for i in range(len(imgs), nrows * ncols):
        axs[i // ncols, i % ncols].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[], )
        for spines in axs[i // ncols, i % ncols].spines.values():
            spines.set_visible(False)
    if save_folder_path:
        plt.savefig(os.path.join(save_folder_path, image_name))
    if display:
        plt.show()
    else:
        plt.close(fig)


class MyFill2(nn.Module):
    """自定义填充模块，将图片变成指定最小宽高的正方形图片，使用黑色填充空白部分。
    如果图片尺寸小于最小宽高，则随机放置原图；否则，居中放置。
    """

    def __init__(self, min_width=0, min_height=0):
        super().__init__()
        self.min_width = min_width
        self.min_height = min_height
        self.__name__ = 'MyFill2(min_width={}, min_height={})'.format(self.min_width, self.min_height)

    def forward(self, x):
        if isinstance(x, Image.Image):
            # 获取原始尺寸
            w, h = x.size
            # 计算目标尺寸
            target_w = max(w, self.min_width) 
            target_h = max(h, self.min_height)
            max_side = max(target_w, target_h)
            # 创建新的背景图像（黑色）
            new_img = Image.new("RGB", (max_side, max_side))
            # 判断是否需要随机放置
            if w < self.min_width or h < self.min_height:
                # 随机生成粘贴位置
                left = np.random.randint(0, max_side - w + 1)
                top = np.random.randint(0, max_side - h + 1)
            else:
                # 随机粘贴，但保证原图完全在新图像内
                left = np.random.randint(0, max_side - w + 1)
                top = np.random.randint(0, max_side - h + 1)
            # 将原图粘贴到背景图像
            new_img.paste(x, (left, top))
            return new_img
        elif isinstance(x, torch.Tensor):
            # 假设输入的张量形状为 [C, H, W]
            c, h, w = x.shape
            target_w = max(w, self.min_width)
            target_h = max(h, self.min_height)
            max_side = max(target_w, target_h)
            # 创建新的张量（黑色背景）
            new_img = torch.zeros(c, max_side, max_side, dtype=x.dtype, device=x.device)
            # 判断是否需要随机放置
            if w < self.min_width or h < self.min_height:
                # 随机生成粘贴位置
                top = torch.randint(0, max_side - h + 1, (1,)).item()
                left = torch.randint(0, max_side - w + 1, (1,)).item()
            else:
                # 随机粘贴，但保证原图完全在新图像内
                left = torch.randint(0, max_side - w + 1, (1,)).item()
                top = torch.randint(0, max_side - h + 1, (1,)).item()
            # 将原图粘贴到背景张量
            new_img[:, top:top + h, left:left + w] = x
            return new_img
        else:
            if debug:
                raise TypeError(f"Unsupported input type: {type(x)}")
            return x
    def __name__(self):
        return 'MyFill2(min_width={}, min_height={})'.format(self.min_width, self.min_height)


import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
import numpy as np
import cv2

class MyFill2_albumentations(ImageOnlyTransform):
    """自定义填充变换，将图片变成指定最小宽高的正方形图片，使用黑色填充空白部分。
    如果图片尺寸小于最小宽高，则随机放置原图；否则，居中放置。
    """

    def __init__(self, min_width=0, min_height=0, always_apply=True, p=1.0):
        super().__init__(always_apply, p)
        self.min_width = min_width
        self.min_height = min_height

    def apply(self, img, **params):
        h, w = img.shape[:2]
        target_w = max(w, self.min_width)
        target_h = max(h, self.min_height)
        max_side = max(target_w, target_h)

        # 创建黑色背景
        if len(img.shape) == 3:
            new_img = np.zeros((max_side, max_side, img.shape[2]), dtype=img.dtype)
        else:
            new_img = np.zeros((max_side, max_side), dtype=img.dtype)

        # 判断是否需要随机放置
        if w < self.min_width or h < self.min_height:
            top = np.random.randint(0, max_side - h + 1)
            left = np.random.randint(0, max_side - w + 1)
        else:
            # 居中放置
            top = (max_side - h) // 2
            left = (max_side - w) // 2

        new_img[top:top + h, left:left + w] = img
        return new_img

    def get_transform_init_args_names(self):
        return ('min_width', 'min_height')

if __name__ == '__main__':
    debug = True
    plt.rcParams["savefig.bbox"] = 'tight'
    torch.manual_seed(1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    # 读取测试图像
    testImages = []
    i = 200000
    testImages_path = os.path.join('data', 'breast', 'cla', 'trainROI')
    for image_name in os.listdir(testImages_path):
        # 要求后缀是png或jpg
        if not image_name.endswith('.png') and not image_name.endswith('.jpg'):
            continue
        image_path = os.path.join(testImages_path, image_name)
        image = Image.open(image_path).convert('RGB')
        # testImages.append(image)
        testImages.append((image_name, image))
        i -= 1
        if i == 0:
            break

    transforms = torch.nn.Sequential(
        MyFill2(256, 256),
        # 其他变换操作
    )

    print(len(testImages))
    for image_name, image in tqdm(testImages):
        transformed_image = transforms(image)
        # print(transformed_image.size)
        show([image, transformed_image], image_name=image_name, save_folder_path=os.path.join('data', 'breast', 'cla', 'trainROI_MyFill2_256'))
