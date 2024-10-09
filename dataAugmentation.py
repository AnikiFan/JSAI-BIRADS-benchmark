from utils.dataAugmentation import Preprocess
from albumentations import Compose, RandomBrightnessContrast
import albumentations as A

if __name__ == '__main__':
    transform = A.Compose([A.Rotate(limit=10, always_apply=True)])
    Preprocess(transform,official_train=False,BUS=False,USG=False,fea_official_train=True).process_image()

    ratio = [2, 1, 3, 4, 5, 6]
    # MixUp(0.4, ratio=ratio).process_image()

    # transform = A.Compose([A.Rotate(limit=10, always_apply=True), A.HorizontalFlip(always_apply=True)])
    # Preprocess(transform, ratio=ratio).process_image()

    # transform = A.Compose([A.Rotate(limit=10, always_apply=True)])
    # Preprocess(transform, ratio=ratio).process_image()

    # transform = A.Compose([A.RandomBrightnessContrast(always_apply=True)])
    # Preprocess(transform, ratio=ratio).process_image()

    transform = A.Compose([A.VerticalFlip(always_apply=True)])
    Preprocess(transform, ratio=ratio).process_image()


    # transform = A.Compose([A.Perspective(scale=(0.05, 0.1), always_apply=True)])
    # Preprocess(transform, ratio=ratio).process_image()

    # transform = A.Compose([A.ElasticTransform(alpha=1, sigma=50, always_apply=True)])
    # Preprocess(transform, ratio=ratio).process_image()

