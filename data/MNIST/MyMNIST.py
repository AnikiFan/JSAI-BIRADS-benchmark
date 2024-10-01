from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Resize, ToTensor
import os


class MyMNIST:
    def __init__(self, data_folder_path, **kwargs):
        self.root = os.path.join(data_folder_path, 'MNIST')
        self.finish = False

    def __iter__(self):
        return self

    def __next__(self):
        if not self.finish:
            self.finish = True
            return (
            MNIST(self.root, train=True, download=True, transform=Compose([Resize((128 , 128)), ToTensor()])),
            MNIST(self.root, train=False, download=True, transform=Compose([Resize((128, 128)), ToTensor()])))
        raise StopIteration
