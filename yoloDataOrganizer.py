import shutil
import os
from tqdm import tqdm
import yaml

'''
Diagnosis assessment

Category 0 Incomplete and needs additional imaging evolution Negative
Category 1 Negative
Category 2 Benign
Category 3 Probably Benign
Category 4 Suspicious
            Catagory 4A: Low suspicion for malignancy (2% to 8% probability of malignancy)
            Catagory 4B: Moderate suspicion for malignancy (9% to 49% probability of malignancy)
            Catagory 4C: High suspicion for malignancy (50% to 95% probability of malignancy)
Category 5 Highly suggestive of malignancy (> 95% probability)
Category 6 Known biopsy-proven malignancy

可能分类策略：
1. 2类、3类、4类(4A类、4B类、4C类合并)、5类、6类
2. Benign(2类、3类)、Suspicious(4类)、Malignant(5类、6类)
'''
def checkAndResumeTxt(txt_path:str,class_name:str):
    '''
    检查txt文件的类别是否正确，如果不正确，则修改为正确的类别
    '''
    class_num = {'2类':1,'3类':2,'4A类':3,'4B类':4,'4C类':5,'5类':6}  #! 注意：官方数据集类别从1开始
    with open(txt_path, 'r') as file:
        lines = file.readlines()

    new_lines = []
    for line in lines:
        elements = line.strip().split()
        if not elements:
            warnings.warn(f'{txt_path} 存在空行')
            continue  # 跳过空行
        if elements[0] != str(class_num[class_name]):
            print(f'{txt_path} 类别错误，应为 {class_num[class_name]}，实际为 {elements[0]}')
            elements[0] = str(class_num[class_name])
            print("已修改")
        new_lines.append(' '.join(elements) + '\n')

    with open(txt_path, 'w') as file:
        file.writelines(new_lines)


def modifyTxt(txt_path:str,
              map_type:['2_3_4A_4B_4C_5','2_3_4_5']
              ):
    '''
    :param txt_path: 需要修改的txt文件路径
    :param map_type: 修改的类别映射类型:
                    '2_3_4A_4B_4C_5'表示2类、3类、4A类、4B类、4C类、5类，
                    '2_3_4_5'表示2类、3类、4类(4A类、4B类、4C类合并)、5类
    '''
    # 不允许txt文件来自breast/cla/official_test
    assert os.path.join(os.getcwd(),'data','breast','cla','official_train') not in txt_path, f"警告！不允许修改官方数据集"
    assert os.path.join(os.getcwd(),'data','breast','cla','official_test') not in txt_path, f"警告！不允许修改官方数据集"
    
    class_num_map = {
        '2_3_4A_4B_4C_5':{
            '1':0, # 2类
            '2':1, # 3类
            '3':2, # 4A类
            '4':3, # 4B类
            '5':4, # 4C类
            '6':5 # 5类
        },
        '2_3_4_5':{
            '1':0, # 2类
            '2':1, # 3类
            '3':2, # 4A类
            '4':2, # 4B类
            '5':2, # 4C类
            '6':3 # 5类
        }
    }
    
    with open(txt_path, 'r') as file:
        lines = file.readlines()

    new_lines = []
    for line in lines:
        elements = line.strip().split()
        elements[0] = str(class_num_map[map_type][elements[0]])
        new_lines.append(' '.join(elements) + '\n')

    with open(txt_path, 'w') as file:
        file.writelines(new_lines)


def organize_data(source_folder:str =os.path.join(os.getcwd(),'data','breast','cla','official_train'),
                  destination_folder:str =os.path.join(os.getcwd(),'data','breast','cla','official_yolo'),
                  type:['train','valid','test']='train',
                  map_type:['2_3_4A_4B_4C_5','2_3_4_5']='2_3_4A_4B_4C_5'
                  ):
    '''
    type: 数据集类型，train, valid, test
    map_type: 类别映射类型，'2_3_4A_4B_4C_5'表示2类、3类、4A类、4B类、4C类、5类，
                                        '2_3_4_5'表示2类、3类、4类(4A类、4B类、4C类合并)、5类
    根据 data/breast/cla/official_train 和 data/breast/cla/official_test 组织数据集
    组织后的数据集格式为：
    destination_folder/images/type/000001.jpg
    destination_folder/labels/type/000001.txt
    '''
    # destination_folder = os.path.join(destination_folder, type)
    assert os.path.exists(source_folder), f"Source folder {source_folder} does not exist"

    for class_folder in os.listdir(source_folder):
        # check 
        if class_folder not in ['2类', '3类', '4A类', '4B类', '4C类', '5类']:
            continue
        class_path = os.path.join(source_folder, class_folder)
        image_folder_path = os.path.join(class_path, 'images') #2类/images
        label_folder_path = os.path.join(class_path, 'labels') #2类/labels

        new_image_folder_path = os.path.join(destination_folder, 'images', type) #data/breast/cla/official_train_yolo/images/type
        new_label_folder_path = os.path.join(destination_folder, 'labels', type) #data/breast/cla/official_train_yolo/labels/type
        
        os.makedirs(new_image_folder_path, exist_ok=True)
        os.makedirs(new_label_folder_path, exist_ok=True)

        for image_file in tqdm(os.listdir(image_folder_path),desc=f'Processing {class_folder} {type} data',leave=False):
            image_file_src = os.path.join(image_folder_path, image_file)
            
            label_file = os.path.splitext(image_file)[0] + '.txt'
            label_file_src = os.path.join(label_folder_path, label_file)
            checkAndResumeTxt(label_file_src,class_folder)
            
            image_file_dst = os.path.join(new_image_folder_path, image_file)
            label_file_dst = os.path.join(new_label_folder_path, label_file)
            shutil.copy(image_file_src, image_file_dst)
            shutil.copy(label_file_src, label_file_dst)
            
            modifyTxt(label_file_dst,map_type) #! 需要修改label文件，整体类别-1，data.yaml中要求从0开始



def getDataYaml(yolo_dst:str= os.path.join(os.getcwd(),'data','breast','cla','official_yoloDetection'),
                map_type:['2_3_4A_4B_4C_5','2_3_4_5']='2_3_4A_4B_4C_5'
                ):
    yaml_content = {
        'path': f'{yolo_dst}',
        'train': 'images/train',
        'val': 'images/val',
        'test': None,
        'names': {}
    }
    
    if map_type == '2_3_4A_4B_4C_5':
        yaml_content['names'] = {
            0: '2',
            1: '3',
            2: '4A',
            3: '4B',
            4: '4C',
            5: '5'
        }
    elif map_type == '2_3_4_5':
        yaml_content['names'] = {
            0: '2',
            1: '3',
            2: '4',
            3: '5'
        }
    
    with open(os.path.join(yolo_dst, 'data.yaml'), 'w') as file:
        yaml.dump(yaml_content, file, allow_unicode=True)
        

def getYoloData(train_src:str= os.path.join(os.getcwd(),'data','breast','cla','official_train'),
                valid_src:str= os.path.join(os.getcwd(),'data','breast','cla','official_test'),
                yolo_dst:str= os.path.join(os.getcwd(),'data','breast','cla','official_yoloDetection'),
                map_type:['2_3_4A_4B_4C_5','2_3_4_5']='2_3_4A_4B_4C_5'
                ):
    assert os.path.exists(train_src), f"Source folder {train_src} does not exist"
    assert os.path.exists(valid_src), f"Source folder {valid_src} does not exist"
    assert not os.path.exists(yolo_dst), f"Destination folder {yolo_dst} already exists"
    organize_data(train_src, yolo_dst, 'train',map_type)
    organize_data(valid_src, yolo_dst, 'val',map_type)
    
    getDataYaml(yolo_dst,map_type)

'''
整合成一个对象
'''

class YoloDataOrganizer:
    def __init__(self,
                 train_src:str =os.path.join(os.getcwd(),'data','breast','cla','official_train'),
                 valid_src:str =os.path.join(os.getcwd(),'data','breast','cla','official_test'),
                 yolo_dst:str =os.path.join(os.getcwd(),'data','breast','cla','official_yoloDetection'),
                 map_type:str ='2_3_4A_4B_4C_5'):
        self.train_src = train_src
        self.valid_src = valid_src
        self.yolo_dst = yolo_dst
        self.map_type = map_type
        self.class_num = {'2类':1,'3类':2,'4A类':3,'4B类':4,'4C类':5,'5类':6}  #! class_num用于对官方数据集进行检查，注意：官方数据集类别从1开始
        self.class_num_map = {
            '2_3_4A_4B_4C_5':{
                '1':0, # 2类
                '2':1, # 3类
                '3':2, # 4A类
                '4':3, # 4B类
                '5':4, # 4C类
                '6':5  # 5类
            },
            '2_3_4_5':{
                '1':0, # 2类
                '2':1, # 3类
                '3':2, # 4A类
                '4':2, # 4B类
                '5':2, # 4C类
                '6':3  # 5类
            }
        }

    def checkAndResumeTxt(self, txt_path:str, class_name:str):
        '''
        检查txt文件的类别是否正确，如果不正确，则修改为正确的类别
        '''
        with open(txt_path, 'r') as file:
            lines = file.readlines()

        new_lines = []
        for line in lines:
            elements = line.strip().split()
            if not elements:
                warnings.warn(f'{txt_path} 存在空行')
                continue  # 跳过空行
            if elements[0] != str(self.class_num[class_name]):
                print(f'{txt_path} 类别错误，应为 {self.class_num[class_name]}，实际为 {elements[0]}')
                elements[0] = str(self.class_num[class_name])
                print("已修改")
            new_lines.append(' '.join(elements) + '\n')

        with open(txt_path, 'w') as file:
            file.writelines(new_lines)

    def modifyTxt(self, txt_path:str):
        '''
        修改txt文件中的类别标签
        '''
        # 不允许txt文件来自官方数据集
        assert os.path.join(os.getcwd(),'data','breast','cla','official_train') not in txt_path, f"警告！不允许修改官方数据集"
        assert os.path.join(os.getcwd(),'data','breast','cla','official_test') not in txt_path, f"警告！不允许修改官方数据集"

        with open(txt_path, 'r') as file:
            lines = file.readlines()

        new_lines = []
        for line in lines:
            elements = line.strip().split()
            elements[0] = str(self.class_num_map[self.map_type][elements[0]])
            new_lines.append(' '.join(elements) + '\n')

        with open(txt_path, 'w') as file:
            file.writelines(new_lines)

    def organize_data(self, source_folder:str, data_type:str):
        '''
        根据给定的源文件夹组织数据集
        '''
        assert os.path.exists(source_folder), f"源文件夹 {source_folder} 不存在"

        for class_folder in os.listdir(source_folder):
            if class_folder not in ['2类', '3类', '4A类', '4B类', '4C类', '5类']:
                continue
            class_path = os.path.join(source_folder, class_folder)
            image_folder_path = os.path.join(class_path, 'images')
            label_folder_path = os.path.join(class_path, 'labels')

            new_image_folder_path = os.path.join(self.yolo_dst, 'images', data_type)
            new_label_folder_path = os.path.join(self.yolo_dst, 'labels', data_type)

            os.makedirs(new_image_folder_path, exist_ok=True)
            os.makedirs(new_label_folder_path, exist_ok=True)

            for image_file in tqdm(os.listdir(image_folder_path), desc=f'Processing {class_folder} {data_type} data', leave=False):
                image_file_src = os.path.join(image_folder_path, image_file)

                label_file = os.path.splitext(image_file)[0] + '.txt'
                label_file_src = os.path.join(label_folder_path, label_file)
                self.checkAndResumeTxt(label_file_src, class_folder)

                image_file_dst = os.path.join(new_image_folder_path, image_file)
                label_file_dst = os.path.join(new_label_folder_path, label_file)
                shutil.copy(image_file_src, image_file_dst)
                shutil.copy(label_file_src, label_file_dst)

                self.modifyTxt(label_file_dst)

    def getDataYaml(self):
        '''
        生成 data.yaml 文件
        '''
        yaml_content = {
            'path': f'{self.yolo_dst}',
            'train': 'images/train',
            'val': 'images/val',
            'test': None,
            'names': {}
        }

        if self.map_type == '2_3_4A_4B_4C_5':
            yaml_content['names'] = {
                0: '2',
                1: '3',
                2: '4A',
                3: '4B',
                4: '4C',
                5: '5'
            }
        elif self.map_type == '2_3_4_5':
            yaml_content['names'] = {
                0: '2',
                1: '3',
                2: '4',
                3: '5'
            }

        with open(os.path.join(self.yolo_dst, 'data.yaml'), 'w') as file:
            yaml.dump(yaml_content, file, allow_unicode=True)

    def getYoloData(self):
        '''
        组织 YOLO 格式的数据集
        '''
        assert os.path.exists(self.train_src), f"训练数据源文件夹 {self.train_src} 不存在"
        assert os.path.exists(self.valid_src), f"验证数据源文件夹 {self.valid_src} 不存在"
        assert not os.path.exists(self.yolo_dst), f"目标文件夹 {self.yolo_dst} 已存在"

        self.organize_data(self.train_src, 'train')
        self.organize_data(self.valid_src, 'val')

        self.getDataYaml()
        
        

if __name__ == '__main__':
    yolo_dst = os.path.join(os.getcwd(),'data','breast','cla','official_yoloDetection_2_3_4_5')
    getYoloData(yolo_dst=yolo_dst,
                map_type='2_3_4_5')
    # getDataYaml()