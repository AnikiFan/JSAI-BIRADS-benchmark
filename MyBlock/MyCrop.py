import os.path
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch
import torchvision.transforms as T
from torchvision.io import read_image
import torchvision.transforms.functional as TF
import random

plt.rcParams["savefig.bbox"] = 'tight'
torch.manual_seed(1)

class MyRotationTransform(nn.Module):
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        super().__init__()
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


def show(imgs):
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = T.ToPILImage()(img.to('cpu'))
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()


test1 = read_image(str(os.path.join(os.pardir, 'data', 'test', '0086.jpg')))
test2 = read_image(str(os.path.join(os.pardir, 'data', 'test', '0088.jpg')))
show([test1, test2])

transforms = torch.nn.Sequential(
    T.RandomCrop(224),
    T.RandomHorizontalFlip(p=0.3),
    MyRotationTransform(angles=[-30, -15, 0, 15, 30]),
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
test1 = test1.to(device)
test2 = test2.to(device)

transformed_test1 = transforms(test1)
transformed_test2 = transforms(test2)
show([transformed_test1, transformed_test2])
