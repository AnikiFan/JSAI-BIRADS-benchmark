import os.path

import matplotlib.pyplot as plt
import numpy as np
import warnings
import torch.nn as nn
import torch
import torchvision.transforms as T
from torchvision.io import read_image
import torchvision.transforms.functional as TF


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
    """Rotate by one of the given angles."""

    def __init__(self):
        super().__init__()

    def __call__(self, x):
        """
        :param x:
        :return: 裁剪去界面后的图像
        裁剪某一行/列的需满足以下条件之一：
        1. 黑色比例超过阈值(黑色这里定义为三通道相同，且都小于等于20)
        2. 白色比例超过阈值(白色这里定义为三通道相同，且都大于等于250)
        3. 彩色比例超过阈值(彩色这里定义为三通道与平均值之差的绝对值与三通道平均值之比之和大于0.5)
        """
        c, h, w = x.shape
        row_tolerance = h * 0.05
        col_tolerance = w * 0.05
        max_row_tolerance_time = 2
        max_col_tolerance_time = 2
        channel_mean = torch.mean(x.to(torch.float), dim=0)
        channel_diff_abs = torch.abs(x - channel_mean).to(torch.uint8)
        channel_is_same = (channel_diff_abs == 0)
        channel_is_black = (x <= 20)
        channel_is_white = (x >= 250)
        channel_is_color = torch.sum(channel_diff_abs, dim=0)/channel_mean > 0.5
        tmp = channel_is_same & (channel_is_white | channel_is_black)
        pure_mask = tmp[0]
        for i in range(1, tmp.size(0)):
            pure_mask = pure_mask | tmp[i]

        pure_row_mask = (torch.sum(pure_mask, dim=1) / w) >= 0.85
        pure_col_mask = (torch.sum(pure_mask, dim=0) / h) >= 0.80
        color_row_mask = (torch.sum(channel_is_color, dim=1) / w) >= 0.5
        color_col_mask = (torch.sum(channel_is_color, dim=0) / h) >= 0.3
        row_mask = pure_row_mask | color_row_mask
        col_mask = pure_col_mask | color_col_mask

        left, right, top, bottom = 0, w - 1, 0, h - 1

        tolerance_cnt = 0
        tolerance_time = 0
        p = left
        while p < right and tolerance_time < col_tolerance:
            if col_mask[p]:
                left = p
                p += 1
                if tolerance_cnt:
                    tolerance_cnt = 0
                    tolerance_time += 1
            else:
                if tolerance_cnt >= col_tolerance:
                    break
                tolerance_cnt += 1
                p += 1

        tolerance_cnt = 0
        tolerance_time = 0
        p = right
        while p > left and tolerance_time < col_tolerance:
            if col_mask[p]:
                right = p
                p -= 1
                if tolerance_cnt:
                    tolerance_cnt = 0
                    tolerance_time += 1
            else:
                if tolerance_cnt >= col_tolerance:
                    break
                tolerance_cnt += 1
                p -= 1

        tolerance_cnt = 0
        tolerance_time = 0
        p = top
        while p < bottom and tolerance_time < row_tolerance:
            if row_mask[p]:
                top = p
                p += 1
                if tolerance_cnt:
                    tolerance_cnt = 0
                    tolerance_time += 1
            else:
                if tolerance_cnt >= row_tolerance:
                    break
                tolerance_cnt += 1
                p += 1

        tolerance_cnt = 0
        tolerance_time = 0
        p = bottom
        while p > top and tolerance_time < row_tolerance:
            if row_mask[p]:
                bottom = p
                p -= 1
                if tolerance_cnt:
                    tolerance_cnt = 0
                    tolerance_time += 1
            else:
                if tolerance_cnt >= row_tolerance:
                    break
                tolerance_cnt += 1
                p -= 1

        if right - left < w * 0.4:
            warnings.warn('width too small after cropped!')
            left, right = 0, w - 1
        if bottom - top < h * 0.4:
            warnings.warn('height too small after cropped!')
            top, bottom = 0, h - 1

        # DEBUG用
        # show([x,TF.crop(x, top, left, bottom - top, right - left)])
        return TF.crop(x, top, left, bottom - top, right - left)




if __name__ == '__main__':
    plt.rcParams["savefig.bbox"] = 'tight'
    torch.manual_seed(1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    testImages = [read_image(os.path.join(os.pardir,os.pardir, 'data', 'test', image)).to(device) for image in
                  os.listdir(os.path.join(os.pardir,os.pardir, 'data', 'test'))]

    transforms = torch.nn.Sequential(
        MyCrop(),
        # T.RandomCrop(224),
        # T.RandomHorizontalFlip(p=0.3),
    )
    for image in testImages:
        transforms(image)