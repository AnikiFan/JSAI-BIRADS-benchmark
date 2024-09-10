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