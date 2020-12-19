

## 人工智能导论实验3代码说明文档—— 分类算法实验

​																																																				——created by 屈德林 in 2020/12/19

目录：

[TOC]

## 简介

在本实验中，我手工实现了所有算法，包括朴素贝叶斯，决策树ID3算法，决策树C4.5算法，决策树CART算法，人工神经网络BP算法，支持向量积SVM的SMO算法，并且对决策树的三个算法决策树结果进行了可视化。统计了所有算法的训练准确率和测试准确率，给出了时间开销。

本文件是对lab3目录下代码的介绍，你将会了解`lab3`目录结构，以及程序运行的方法。

## 运行环境

- 操作系统：`Ubuntu20.04 LST`
- 解释器：python



## 文件目录

```bash
.
├── data
│   ├── dataset.txt
│   ├── predict.txt
│   └── test.txt
├── DecisionTree
│   ├── C45_Decision.py
│   ├── CART_Decision.py
│   ├── fig
│   │   ├── C45.png
│   │   ├── CART.png
│   │   └── ID3.png
│   ├── ID3_Decision.py
│   ├── Sklearn_Decision.py
│   └── utils
│       └── plotDecisionTree.py
├── NaiveBayes.py
├── NeuralNetwork.py
├── README.md
├── sklearns
│   ├── Classification.py
│   ├── dataset.txt
│   ├── predict.txt
│   └── test.txt
└── SVM.py
```

- `data`数据集
- `DecisionTree`：决策树算法目录
- `ID3_Decision.py`：ID3算法求解决策树
- `C45_Decision.py`：C45算法求解决策树
- `CART_Decision.py`：CART算法求解决策树
- `fig`：决策树可视化结果
- `utils`：可视化工具类
- `NaiveBayes.py`：朴素贝叶斯
- `NeuralNetwork.py`：BP算法神经网络
- `SVM.py`：SVM算法求解
- `sklearns`：使用机器学习sklearn库
- `README.md`：说明文档



## 如何运行？

- 使用python解释器运行不同算法

```bash
$ python NaiveBayes.py 
$ python NeuralNetwork.py 
$ python SVM.py 
$ python ID3_Decision.py 
$ python C45_Decision.py 
$ python CART_Decision.py 
```

![image-20201219120515124](https://i.loli.net/2020/12/19/DkQjqyRntzJHa87.png)