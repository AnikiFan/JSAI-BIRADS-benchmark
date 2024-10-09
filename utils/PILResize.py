from PIL import Image
from typing import Union, Tuple

class PILResize:
    def __init__(self, size: Union[int, Tuple[int, int]], interpolation: int = Image.BILINEAR):
        """
        参数：
            size (int 或 tuple)：期望的输出尺寸。如果size是一个tuple，
                                 输出尺寸将匹配这个值。如果size是一个整数，
                                 图像的较小边将被调整到这个数值，保持宽高比。
            interpolation (int, 可选)：期望的插值方法。默认是Image.BILINEAR。
        """
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img: Image) -> Image:
        """
        参数：
            img (PIL Image)：需要调整大小的图像。

        返回：
            PIL Image：调整大小后的图像。
        """
        return img.resize(self.size, self.interpolation)

# # 在torchvision.transforms中使用示例
# import torchvision.transforms as transforms

# transform = transforms.Compose([
#     PILResize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])
