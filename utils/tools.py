import torch
import torchvision.transforms as transforms
import os

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


def save_checkpoint(state, checkpoint_dir, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint_dir, filename)
    relative_filepath = os.path.relpath(filepath)
    print(f"=> saving checkpoint at '{relative_filepath}'")
    torch.save(state, filepath)

def load_checkpoint(model, optimizer, checkpoint_path):
    if os.path.isfile(checkpoint_path):
        print(f"=> loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        best_vloss = checkpoint['best_vloss']
        print(f"=> loaded checkpoint '{checkpoint_path}' (epoch {epoch})")
        print(f"-- current epoch: {epoch}")
        print(f"-- current lr: {optimizer.param_groups[0]['lr']}")
        print(f"-- current best_vloss: {best_vloss}")
        return model, optimizer, epoch, best_vloss
    else:
        print(f"=> no checkpoint found at '{checkpoint_path}'")
        return model, optimizer, 0, float('inf')
    
    
