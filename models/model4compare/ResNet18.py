from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights, resnet50, ResNet50_Weights, \
    resnet101, ResNet101_Weights, resnet152, ResNet152_Weights
import torch


def ResNet18(num_classes=10, **kwargs):
    pretrained_net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    pretrained_net.fc = torch.nn.Linear(pretrained_net.fc.in_features, num_classes)
    torch.nn.init.xavier_uniform_(pretrained_net.fc.weight)
    return pretrained_net
