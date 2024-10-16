import os.path
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch
import torchvision.transforms as T
from torchvision.io import read_image
import torchvision.transforms.functional as TF
from PIL import Image  # 加入PIL用于处理输入MyCrop是PIL Image而不是Tensor的情况

debug = False


def show(imgs):
    ncols, nrows = (len(imgs) + 1) // 2, 2
    fix, axs = plt.subplots(ncols=ncols, nrows=nrows, squeeze=False)
    for i, img in enumerate(imgs):
        if isinstance(img, torch.Tensor):
            img = T.ToPILImage()(img.to('cpu'))
        axs[i // ncols, i % ncols].imshow(np.asarray(img))
        axs[i // ncols, i % ncols].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    for i in range(len(imgs), nrows * ncols):
        axs[i // ncols, i % ncols].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[], )
        for spines in axs[i // ncols, i % ncols].spines.values():
            spines.set_visible(False)
    plt.show()


class MyCrop(nn.Module):
    """自定义裁剪模块，将图片变成宽高一致的正方形图片，使用黑色填充空白部分。"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        if isinstance(x, Image.Image):
            # 获取原始尺寸
            w, h = x.size
            # 计算目标尺寸
            max_side = max(w, h)
            # 创建新的背景图像（黑色）
            new_img = Image.new("RGB", (max_side, max_side))
            # 将原图粘贴到背景图像的中央
            new_img.paste(x, ((max_side - w) // 2, (max_side - h) // 2))
            return new_img
        elif isinstance(x, torch.Tensor):
            # 假设输入的张量形状为 [C, H, W]
            c, h, w = x.shape
            max_side = max(w, h)
            # 创建新的张量（黑色背景）
            new_img = torch.zeros(c, max_side, max_side, dtype=x.dtype, device=x.device)
            # 计算粘贴位置
            top = (max_side - h) // 2
            left = (max_side - w) // 2
            new_img[:, top:top + h, left:left + w] = x
            return new_img
        else:
            if debug:
                raise TypeError(f"Unsupported input type: {type(x)}")
            return x


if __name__ == '__main__':
    debug = True
    plt.rcParams["savefig.bbox"] = 'tight'
    torch.manual_seed(1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 读取测试图像
    testImages = []
    for image_name in os.listdir(os.path.join('data', 'breast', 'cla', 'train')):
        image_path = os.path.join('data', 'test', image_name)
        image = Image.open(image_path).convert('RGB')
        testImages.append(image)

    transforms = torch.nn.Sequential(
        MyCrop(),
        # 其他变换操作
    )

    for image in testImages:
        transformed_image = transforms(image)
        show([image, transformed_image])
