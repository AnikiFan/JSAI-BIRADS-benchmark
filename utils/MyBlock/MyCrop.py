import os.path
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch
import torchvision.transforms as T
from torchvision.io import read_image
import torchvision.transforms.functional as TF
from ..removeFrame import removeFrame
from PIL import Image  # 加入PIL用于处理输入MyCrop是PIL Image而不是Tensor的情况

debug = False


def show(imgs):
    ncols, nrows = (len(imgs) + 1) // 2, 2
    fix, axs = plt.subplots(ncols=ncols, nrows=nrows, squeeze=False)
    for i, img in enumerate(imgs):
        img = T.ToPILImage()(img.to('cpu'))
        axs[i // ncols, i % ncols].imshow(np.asarray(img))
        axs[i // ncols, i % ncols].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    for i in range(len(imgs), nrows * ncols):
        axs[i // ncols, i % ncols].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[], )
        for spines in axs[i // ncols, i % ncols].spines.values():
            spines.set_visible(False)
    plt.show()


class MyCrop(nn.Module):
    """自定义裁剪模块，能够处理 PIL Image 和 torch.Tensor。"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        if isinstance(x, Image.Image):  # 如果输入是PIL Image，则先转换为Tensor进行剪裁，再转换回PIL Image，给下一个模块使用
            # print("MyCrop Process PIL Image")
            # 将 PIL Image 转换为张量
            x = TF.to_tensor(x)
            # 调用内部裁剪逻辑
            top, left, h, w = removeFrame(x.numpy())
            # 将张量转换回 PIL Image
            x = TF.to_pil_image(x)
        elif isinstance(x, torch.Tensor):
            # 直接处理张量
            top, left, h, w = removeFrame(x.numpy())
        else:
            raise TypeError(f"Unsupported input type: {type(x)}")
        return TF.crop(x, top, left, h, w)


if __name__ == '__main__':
    debug = True
    plt.rcParams["savefig.bbox"] = 'tight'
    torch.manual_seed(1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # note:这里的读取方式和我们train中读取数据集的方式不同，读取数据集的时候使用的是torchvision.datasets.ImageFolder
    testImages = [read_image(os.path.join('data', 'test', image)).to(device) for image in
                  os.listdir(os.path.join('data', 'test'))]

    transforms = torch.nn.Sequential(
        MyCrop(),
        # T.RandomCrop(224),
        # T.RandomHorizontalFlip(p=0.3),
    )
    for image in testImages:
        transforms(image)
        # show([image])
