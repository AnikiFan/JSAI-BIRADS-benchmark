# TDS-Net网络结构

```
      input:1
        |
        |
   FeatureBlock:64
       |
       |------cb1:64
       |      /|
7*64  db1   /  |
       |  /    |
512    |       |
7*128 db2     cb2:128
       |------/|
1024   |       |
7*128 db3     cb3:128
       |------/|
1024   |       |
7*128 db4     cb4:128
       |      /
       |    /
   Classifier:1
       |
       |
     output
```

# 仓库目录结构

```
D:.
├─.idea
│  └─inspectionProfiles
├─data
│  ├─breast
│  │  ├─.ipynb_checkpoints
│  │  ├─.jupyter
│  │  │  └─desktop-workspaces
│  │  ├─test_A
│  │  │  ├─cla
│  │  │  │  ├─2类
│  │  │  │  │  ├─images
│  │  │  │  │  └─labels
│  │  │  │  ├─3类
│  │  │  │  │  ├─images
│  │  │  │  │  └─labels
│  │  │  │  ├─4A类
│  │  │  │  │  ├─images
│  │  │  │  │  └─labels
│  │  │  │  ├─4B类
│  │  │  │  │  ├─images
│  │  │  │  │  └─labels
│  │  │  │  ├─4C类
│  │  │  │  │  ├─images
│  │  │  │  │  └─labels
│  │  │  │  └─5类
│  │  │  │      ├─images
│  │  │  │      └─labels
│  │  │  └─fea
│  │  │      ├─boundary_labels
│  │  │      ├─calcification_labels
│  │  │      ├─direction_labels
│  │  │      ├─images
│  │  │      └─shape_labels
│  │  ├─train
│  │  │  ├─cla
│  │  │  │  ├─2类
│  │  │  │  │  ├─images
│  │  │  │  │  └─labels
│  │  │  │  ├─3类
│  │  │  │  │  ├─images
│  │  │  │  │  └─labels
│  │  │  │  ├─4A类
│  │  │  │  │  ├─images
│  │  │  │  │  └─labels
│  │  │  │  ├─4B类
│  │  │  │  │  ├─images
│  │  │  │  │  └─labels
│  │  │  │  ├─4C类
│  │  │  │  │  ├─images
│  │  │  │  │  └─labels
│  │  │  │  └─5类
│  │  │  │      ├─images
│  │  │  │      └─labels
│  │  │  └─fea
│  │  │      ├─boundary_labels
│  │  │      ├─calcification_labels
│  │  │      ├─direction_labels
│  │  │      ├─images
│  │  │      └─shape_labels
│  │  └─train_valid_test
│  │      ├─test
│  │      │  └─unknown
│  │      ├─train
│  │      │  ├─2类
│  │      │  ├─3类
│  │      │  ├─4A类
│  │      │  ├─4B类
│  │      │  ├─4C类
│  │      │  └─5类
│  │      ├─train_valid
│  │      │  ├─2类
│  │      │  ├─3类
│  │      │  ├─4A类
│  │      │  ├─4B类
│  │      │  ├─4C类
│  │      │  └─5类
│  │      └─valid
│  │          ├─2类
│  │          ├─3类
│  │          ├─4A类
│  │          ├─4B类
│  │          ├─4C类
│  │          └─5类
│  └─FashionMNIST
│      └─raw
├─model4compare
│  └─__pycache__
├─MyBlock
├─TDSNet
└─__pycache__
```




# train保存的模型文件结构
```
checkPoint
.
├── Unet_Breast_20240911_163649
│   ├── cfg.json
│   ├── model
│   │   ├── Unet_ac0.386731_f10.232353_0.pth
│   │   └── Unet_ac0.422735_f10.289592_1.pth
│   ├── model_cfg.json
│   ├── optimizer
│   │   ├── Unet_ac0.386731_f10.232353_0.pth
│   │   └── Unet_ac0.422735_f10.289592_1.pth
│   └── transforms_cfg.json
└── readme.txt
```


# 新增文件夹说明：
```
**checkPoint** ： 保存训练过程中的模型文件
model_data : Unet中用来下载预训练模型的地址，建议其他模型与之一致
**models** ： 编写的模型文件，建议把TDSNet也放进去
utils ： 一些工具函数，如检查数据集，早停，输出过滤（过滤f1wanning）等
```
