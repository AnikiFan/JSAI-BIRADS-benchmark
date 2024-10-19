from ultralytics import YOLO
import os

model_name = 'yolo11s.pt'
model = YOLO(model_name)



data_path = os.path.join('data', 'breast', 'cla', 'official_yoloDetection')


mode = 'train'

if __name__ == '__main__':
    if mode == 'train':
        results = model.train(data=os.path.join(data_path, 'data.yaml'),
                      epochs=1000, imgsz=640, cache="ram",
                      save = True,
                      project = 'yoloDetection',
                      name = str(model_name).split('.')[0],
                      workers=10, 
                      batch=0.9, # 使用0.9的GPU显存
                    #   hyp = os.path.join('hyp.yaml')
                      )
    elif mode == 'predict':
        # /mnt/AIC/DLApproach/data/breast/cla/official_yoloDetection/images/val/0121.jpg
        img_path = os.path.join(data_path, 'images', 'val', '0121.jpg')
        results = model.predict(source=img_path,
                                save = True,
                                project = 'yoloDetection',
                                name = str(model_name).split('.')[0],
                                save_txt = True,
                                save_conf = True,
                                save_crop = True,
                                visualize = True,
                                show = True,
                                )
    

