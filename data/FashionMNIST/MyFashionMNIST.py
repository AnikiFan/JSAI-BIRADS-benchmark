from torchvision.datasets import FashionMNIST
from torchvision.transforms import Compose, Resize, ToTensor
import os


class MyFashionMNIST:
    def __init__(self, data_folder_path, **kwargs):
        self.root = os.path.join(data_folder_path, 'FashionMNIST')
        self.finish = False

    def __iter__(self):
        return self

    def __next__(self):
        if not self.finish:
            self.finish = True
            return (
            FashionMNIST(self.root, train=True, download=True, transform=Compose([Resize((224, 224)), ToTensor()])),
            FashionMNIST(self.root, train=False, download=True, transform=Compose([Resize((224, 224)), ToTensor()])))
        raise StopIteration
