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
