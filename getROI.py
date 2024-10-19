import os
import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from torchvision.io import read_image
from tqdm import tqdm

def parse_label_file(file_path:str,line_limit:int=None)->list[dict]:
    """
    解析标签文件并返回数据列表。
    每个标签包含类别、中心点x、中心点y、宽度和高度。
    :param file_path: 标签文件的路径
    :return: 包含解析后数据的列表
    """
    labels = []
    line_count = 0
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) != 5:
                continue  # 跳过格式不正确的行
            label = {
                'class': int(parts[0]),
                'center_x': float(parts[1]),
                'center_y': float(parts[2]),
                'width': float(parts[3]),
                'height': float(parts[4])
            }
            labels.append(label)
            line_count += 1
            if line_limit and line_count >= line_limit:
                break
    return labels

def getOfficialTrainGroundTruthWithROI(official_train_path:str=os.path.join(os.curdir,'data','breast','cla','official_train'),
                                       save:bool=True,
                                       official_train_ground_truth_ROI_dst:str=os.path.join(os.curdir,'data','breast','cla','official_train','ground_truth_ROI.csv')
                                             )->pd.DataFrame:
    """
    获取官方训练集的ground truth，并包含ROI
    """
    if os.path.exists(official_train_ground_truth_ROI_dst):
        print(f"official_train_ground_truth_ROI_dst: {official_train_ground_truth_ROI_dst} 已存在,读取中...")
        official_train_ground_truth_ROI = pd.read_csv(official_train_ground_truth_ROI_dst)
        official_train_ground_truth_ROI.columns = ["file_name","label","x","y","width","height"]
        return official_train_ground_truth_ROI

    official_train_images_path = [os.path.join(official_train_path,className,'images')
            for className in os.listdir(official_train_path) if className in ['2类','3类','4A类','4B类','4C类','5类'] ]
    official_train_labels_path = [os.path.join(official_train_path,className,'labels')
            for className in os.listdir(official_train_path) if className in ['2类','3类','4A类','4B类','4C类','5类'] ]

    official_train_ground_truth_records = []
    # 读取official_train_lables
    for label_path in official_train_labels_path:
        for label_file in tqdm(os.listdir(label_path)):
            with open(os.path.join(label_path,label_file),'r') as f:
                parsed_labels = parse_label_file(os.path.join(label_path,label_file),line_limit=1)
            official_train_ground_truth_records.append({"file_name":str(label_file),"label":str(parsed_labels[0]['class']-1), #此处需要-1因为官方的txt是从1开始分配
                                                        "x":parsed_labels[0]['center_x'],"y":parsed_labels[0]['center_y'],
                                                        "width":parsed_labels[0]['width'],"height":parsed_labels[0]['height']})
    official_train_ground_truth_ROI = pd.DataFrame(official_train_ground_truth_records)
    official_train_ground_truth_ROI.columns = ["file_name","label","x","y","width","height"]
    
    if save:
        assert not os.path.exists(official_train_ground_truth_ROI_dst),f"official_train_ground_truth_ROI_dst: {official_train_ground_truth_ROI_dst} 已存在"
        official_train_ground_truth_ROI.to_csv(official_train_ground_truth_ROI_dst,index=False)
    return official_train_ground_truth_ROI


def add_ROI_to_ground_truth(ground_truth_csv_src:str=os.path.join(os.curdir,'data','breast','cla','train','ground_truth.csv'),
                            ROI_df:pd.DataFrame=getOfficialTrainGroundTruthWithROI(os.path.join(os.curdir,'data','breast','cla','official_train')),
                            save:bool=True,
                            )->pd.DataFrame:
    '''
    将ROI添加到ground truth中
    :@param ground_truth_csv_src: ground truth的csv文件路径
    :@param ROI_df: ROI的DataFrame,默认使用getOfficialTrainGroundTruthWithROI()获取的官方训练集的ground truth
    :@param save: 是否保存到csv文件
    :@return: 添加了ROI的ground truth的DataFrame
    
    :添加了ROI的ground_truth_ROI.csv的保存路径，默认在与ground_truth_csv_src同一目录下
    '''
    assert os.path.exists(ground_truth_csv_src),f"ground_truth_csv_src: {ground_truth_csv_src} 不存在"
    ground_truth = pd.read_csv(ground_truth_csv_src)
    ground_truth_ROI = ground_truth.copy()
    # 合并前确保两个数据框中的 'label' 列都是字符串类型
    ground_truth_ROI['label'] = ground_truth_ROI['label'].astype(str)
    ROI_df['label'] = ROI_df['label'].astype(str)
    # 合并两个数据框，即根据official_train_ground_truth_ROI填写输入的ground_truth_csv_src的x,y,width,height
    ground_truth_ROI = ground_truth_ROI.merge(
        ROI_df[['file_name', 'label', 'x', 'y', 'width', 'height']],
        on=['file_name','label'],
        how='inner' # 此处需要inner因为train中可能有来自BUSI的图片，而BUSI的图片没有ROI
        )
    # 将ground_truth_csv_src的目录名作为ground_truth_csv_dst的目录名
    if save:
        ground_truth_csv_dst = os.path.join(os.path.dirname(ground_truth_csv_src),'ground_truth_ROI.csv')
        # assert not os.path.exists(ground_truth_csv_dst),f"ground_truth_csv_dst: {ground_truth_csv_dst} 已存在"
        ground_truth_ROI.to_csv(ground_truth_csv_dst,index=False)
    return ground_truth_ROI





def cropImageWithROI(image:Image.Image,x:float,y:float,width:float,height:float,ratio:float=1)->Image.Image:
    '''
    根据x,y,width,height裁剪图像
    :@param ratio: 裁剪图像的比例，默认为1
    '''
    image_width,image_height = image.size
    # 将归一化坐标转换为像素坐标
    x_pixel = int(x * image_width)
    y_pixel = int(y * image_height)
    width_pixel = int(width * image_width)
    height_pixel = int(height * image_height)
    
    x_left = max(x_pixel - width_pixel/2*ratio, 0)
    y_upper = max(y_pixel - height_pixel/2*ratio, 0)
    x_right = min(x_pixel + width_pixel/2*ratio, image_width)
    y_lower = min(y_pixel + height_pixel/2*ratio, image_height)

    # 裁剪图像
    cropped_image = image.crop((x_left, y_upper, x_right, y_lower))
    return cropped_image

def getROIImageSet(imageSet_src:str,
                   imageSet_dst:str,
                   save:bool=False,
                   overwrite:bool=False,
                   ratio:float=1
                   ):
    """
    处理图像集并保存裁剪后的ROI图像。
    :param imageSet_src: 源图像集路径
    :param imageSet_dst: 目标图像集路径
    :param overwrite: 是否覆盖现有目标文件夹
    """
    assert os.path.exists(imageSet_src), f"imageSet_src: {imageSet_src} 不存在"
    if os.path.exists(imageSet_dst):
        if overwrite:
            print(f"警告：目标文件夹 {imageSet_dst} 将被覆盖")
        else:
            raise FileExistsError(f"imageSet_dst: {imageSet_dst} 已存在。如果要覆盖，请设置 overwrite=True")
    os.makedirs(imageSet_dst, exist_ok=overwrite)
    # 假设这个函数返回一个DataFrame，包含文件名和ROI信息
    ground_truth_ROI = add_ROI_to_ground_truth(os.path.join(imageSet_src, 'ground_truth.csv'), save=False)
    for image_name in tqdm(ground_truth_ROI['file_name'],desc=f"Processing with ratio {ratio}"): # 此处需要ground_truth_ROI['file_name']因为train中可能有来自BUSI的图片，而BUSI的图片没有ROI
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                # 获取ROI信息
                roi_info = ground_truth_ROI.loc[ground_truth_ROI['file_name'] == image_name, 'x':'height'].values[0]
                assert len(roi_info) == 4,f"图像 {image_name} 的ROI信息不完整"
                
                x, y, width, height = roi_info
                
                # 打开并裁剪图像
                with Image.open(os.path.join(imageSet_src, image_name)) as img:
                    cropped_image = cropImageWithROI(img, x, y, width, height,ratio=ratio)
                    if save:
                        cropped_image.save(os.path.join(imageSet_dst, image_name))
            except Exception as e:
                print(f"处理图像 {image_name} 时出错：{str(e)}")
    # 同理在这个文件夹中保存ground_truth.csv（只有filename和label）和ground_truth_ROI.csv
    ground_truth = ground_truth_ROI[['file_name','label']]
    # 将label列中的字符串转换为列表
    ground_truth['label'] = ground_truth['label'].apply(lambda x: int(x))
    ground_truth_ROI['label'] = ground_truth_ROI['label'].apply(lambda x: int(x))
    ground_truth.info()
    ground_truth_ROI.info()
    ground_truth.to_csv(os.path.join(imageSet_dst, 'ground_truth.csv'), index=False)
    ground_truth_ROI.to_csv(os.path.join(imageSet_dst, 'ground_truth_ROI.csv'), index=False)

    print("ROI图像集处理完成")


if __name__ == '__main__':

# getROIImageSet(os.path.join(os.curdir,'data','breast','cla','train'),
#                os.path.join(os.curdir,'data','breast','cla','trainROI'),
#                overwrite=True,
#                save=True,
#                ratio=2)

    # getROIImageSet(os.path.join(os.curdir,'data','breast','cla','train'),
    #             os.path.join(os.curdir,'data','breast','cla','trainROI_2'),
    #             overwrite=False,
    #             save=True,
    #             ratio=2)
    
    ratio_list = [
                #   2,
                  2.5,
                  #3,
                  3.5,
                #   4,
                  4.5,
                #   5,
                  5.5,
                #   6,
                  6.5,
                #   7
                  ]

    for ratio in ratio_list:
        getROIImageSet(os.path.join(os.curdir,'data','breast','cla','train'),
                os.path.join(os.curdir,'data','breast','cla','trainROI_'+str(ratio)),
                overwrite=False,
                save=True,
                ratio=ratio)
        
