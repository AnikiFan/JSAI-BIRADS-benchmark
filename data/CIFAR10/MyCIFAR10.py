from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Resize, ToTensor
import os


class MyCIFAR10:
    def __init__(self, data_folder_path, **kwargs):
        self.root = os.path.join(data_folder_path, 'CIFAR10')
        self.finish = False

    def __iter__(self):
        return self

    def __next__(self):
        if not self.finish:
            self.finish = True
            return (
            CIFAR10(self.root, train=True, download=True, transform=Compose([Resize((128, 128)), ToTensor()])),
            CIFAR10(self.root, train=False, download=True, transform=Compose([Resize((128, 128)), ToTensor()])))
        raise StopIteration
