import numpy as np
from logging import debug
def removeFrame(x:np.ndarray)->Tuple[int,int,int,int]:
    """
    对张量进行裁剪。
    :param x: 输入ndarray，形状为 (C, H, W)
    :return: 裁剪后的图像在原图像中的位置(top,left,h,w)
    裁剪某一行/列的需满足以下条件之一：
    1. 黑色比例超过阈值(黑色这里定义为三通道相同，且都小于等于20)
    2. 白色比例超过阈值(白色这里定义为三通道相同，且都大于等于250)
    3. 彩色比例超过阈值(彩色这里定义为三通道与平均值之差的绝对值与三通道平均值之比之和大于0.5)
    """
    c, h, w = x.shape
    assert c == 3 or c == 1,"input shape should be (c,h,w)!"
    row_tolerance = h * 0.05
    col_tolerance = w * 0.05
    max_row_tolerance_time = 2
    max_col_tolerance_time = 2

    channel_mean = np.mean(x.astype(np.float_), axis=0)
    channel_diff_abs = np.abs(x - channel_mean).astype(np.uint8)
    channel_is_same = (channel_diff_abs == 0)
    channel_is_black = (x <= 20)
    channel_is_white = (x >= 250)
    # To avoid division by zero, add a small epsilon
    epsilon = 1e-8
    channel_is_color = (np.sum(channel_diff_abs, axis=0) / (channel_mean + epsilon)) > 0.5

    tmp = channel_is_same & (channel_is_white | channel_is_black)
    pure_mask = tmp[0]
    for i in range(1, tmp.shape[0]):
        pure_mask = pure_mask | tmp[i]
    pure_row_mask = (np.sum(pure_mask, axis=1) / w) >= 0.85
    pure_col_mask = (np.sum(pure_mask, axis=0) / h) >= 0.80
    color_row_mask = (np.sum(channel_is_color, axis=1) / w) >= 0.5
    color_col_mask = (np.sum(channel_is_color, axis=0) / h) >= 0.3
    row_mask = pure_row_mask | color_row_mask
    col_mask = pure_col_mask | color_col_mask

    left, right, top, bottom = 0, w - 1, 0, h - 1

    # 左边界
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

    # 右边界
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

    # 上边界
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

    # 下边界
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

    if (right - left) < w * 0.4:
        debug('width too small after cropped!')
        left, right = 0, w - 1
    if (bottom - top) < h * 0.4:
        debug('height too small after cropped!')
        top, bottom = 0, h - 1

    return top,left,bottom-top,right-left
