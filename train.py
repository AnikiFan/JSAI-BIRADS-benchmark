import torch
import torchvision
from torch import nn
from d2l import torch as d2l
from TDSNet import TDSNet
if __name__ == '__main__':
    model = TDSNet(lr=0.01)
    data = d2l.FashionMNIST(batch_size=128, resize=(224, 224))
    trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
    trainer.fit(model, data)
