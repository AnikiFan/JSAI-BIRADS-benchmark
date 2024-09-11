import torch
import torchvision.transforms as transforms

# 检查设备并设置 device
def getDevice():
    if torch.cuda.is_available():
        # device = torch.device("cuda")  # 使用 CUDA (GPU)
        device = "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        # device = torch.device("mps")   # 使用 MPS (Apple Silicon)
        device = "mps"
    else:
        # device = torch.device("cpu")   # 使用 CPU
        device = "cpu"
    # print(f"Using device: {device}")
    return device


# 根据配置创建 transforms
def create_transforms(transform_config):
    transform_list = []
    for transform_name, params in transform_config.items():
        transform = getattr(transforms, transform_name)(**params)
        transform_list.append(transform)
    return transforms.Compose(transform_list)

