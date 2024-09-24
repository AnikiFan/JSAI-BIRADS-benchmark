import pandas as pd
from warnings import warn
import os
from tqdm import tqdm
import shutil


def check_label(ground_truth, base_path, file_name):
    with open(os.path.join(base_path, 'labels', file_name), 'r') as file:
        content = file.readlines()
        if len(content) != 1:
            warn(f"multiple label:{os.path.join(base_path, 'labels', file_name)}")
            return False
        if len(content[0].split()) != 5:
            warn(f"invalid label:{os.path.join(base_path, 'labels', file_name)}")
            return False
        if content[0].split()[0] != str(ground_truth):
            warn(f"incompatible label:{os.path.join(base_path, 'labels', file_name)}")
            return False
    return True


def move_image(image_path, label_name, dst):
    png_suffix = label_name.replace('txt', 'png')
    jpg_suffix = label_name.replace('txt', 'jpg')
    try:
        shutil.copy(os.path.join(image_path, png_suffix), dst)
    except FileNotFoundError:
        try:
            shutil.copy(os.path.join(image_path, jpg_suffix), dst)
        except FileNotFoundError:
            warn(f"image {image_path+label_name.replace('.txt')} doesn't exist!")


data_path = os.path.join(os.pardir, 'data', 'breast', 'test_A', 'cla')
dst = os.path.join(os.pardir, 'data', 'breast', 'testA', 'cla')
os.makedirs(dst, exist_ok=True)
label_tables = []
for label, folder in tqdm(enumerate(os.listdir(data_path))):
    valid_labels = list(filter(lambda x: check_label(label + 1, os.path.join(data_path, folder), x),
                               os.listdir(os.path.join(data_path, folder, 'labels'))))
    list(map(lambda x: move_image(os.path.join(data_path, folder, 'images'), x, dst), valid_labels))
    label_table = pd.DataFrame({'file_name': valid_labels})
    label_table['label'] = label
    label_tables.append(label_table)
pd.concat(label_tables, axis=0).reset_index(drop=True).to_csv(os.path.join(dst, 'ground_true.csv'))
