import os
from PIL import Image
from tqdm import tqdm

def get_max_image_size(folder_path):
    max_width = 0
    max_height = 0
    
    for filename in tqdm(os.listdir(folder_path)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            img_path = os.path.join(folder_path, filename)
            with Image.open(img_path) as img:
                width, height = img.size
                max_width = max(max_width, width)
                max_height = max(max_height, height)
    
    return max_width, max_height

def main():
    folder_path = os.path.join(os.getcwd(),'data','breast','cla','trainROI')       #/mnt/AIC/DLApproach/data/breast/cla/trainROI
    max_width, max_height = get_max_image_size(folder_path)
    print(f"文件夹中最大图片尺寸为: 宽度 {max_width} 像素, 高度 {max_height} 像素")

if __name__ == "__main__":
    main()
