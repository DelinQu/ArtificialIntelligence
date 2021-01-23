

## 人工智能导论实验4代码说明文档—— 深度学习算法及应用

​																																																				——created by 屈德林 in 2021/01/09

目录：

[TOC]

## 简介

在本实验中，我

本文件是对lab3目录下代码的介绍，你将会了解`lab3`目录结构，以及程序运行的方法。

## 运行环境

硬件：计算机
软件：操作系统：Linux 
应用软件：PyTorch，CUDA , Python，使用CNN进行图像分类识别



## 文件目录

```bash
$ tree
.
├── cnn.pt
├── data
│   └── MNIST
│       ├── processed
│       │   ├── test.pt
│       │   └── training.pt
│       └── raw
│           ├── t10k-images-idx3-ubyte
│           ├── t10k-images-idx3-ubyte.gz
│           ├── t10k-labels-idx1-ubyte
│           ├── t10k-labels-idx1-ubyte.gz
│           ├── train-images-idx3-ubyte
│           ├── train-images-idx3-ubyte.gz
│           ├── train-labels-idx1-ubyte
│           └── train-labels-idx1-ubyte.gz
├── fig
│   ├── Batch.png
│   ├── one.png
│   └── preBatch.png
├── MINST.py
├── mnist_net.pth
└── README.md
```

- `data`数据集
- `fig`：可视化结果
- `MINST.py`：CNN网络分类MINST数据集



## 如何运行？

- 使用python解释器运行

```zsh
$ python MINST.py
```

![image-20210109224554662](https://i.loli.net/2021/01/09/S3G6T5rIUHb79gW.png) 