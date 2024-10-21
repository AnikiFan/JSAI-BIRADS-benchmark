from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights, resnet50, ResNet50_Weights, \
    resnet101, ResNet101_Weights, resnet152, ResNet152_Weights
import torch


def ResNet18(num_classes, **kwargs):
    pretrained_net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    pretrained_net.fc = torch.nn.Linear(pretrained_net.fc.in_features, num_classes)
    torch.nn.init.xavier_uniform_(pretrained_net.fc.weight)
    return pretrained_net

def ResNet34(num_classes, **kwargs):
    pretrained_net = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    pretrained_net.fc = torch.nn.Linear(pretrained_net.fc.in_features, num_classes)
    torch.nn.init.xavier_uniform_(pretrained_net.fc.weight)
    return pretrained_net

def ResNet50(num_classes, **kwargs):
    pretrained_net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    pretrained_net.fc = torch.nn.Linear(pretrained_net.fc.in_features, num_classes)
    torch.nn.init.xavier_uniform_(pretrained_net.fc.weight)
    return pretrained_net

def ResNet101(num_classes, **kwargs):
    pretrained_net = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
    pretrained_net.fc = torch.nn.Linear(pretrained_net.fc.in_features, num_classes)
    torch.nn.init.xavier_uniform_(pretrained_net.fc.weight)
    return pretrained_net

def ResNet152(num_classes, **kwargs):
    pretrained_net = resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)
    pretrained_net.fc = torch.nn.Linear(pretrained_net.fc.in_features, num_classes)
    torch.nn.init.xavier_uniform_(pretrained_net.fc.weight)
    return pretrained_net
