## 人工智能导论实验2—— Prolog编程求解图搜索问题

![image-20201119162620058](README.assets/image-20201119162620058.png)

​																																																				——created by 屈德林 in 2020/11/16

目录：

[TOC]

## 简介

- 本文件是对lab2目录下传道士与野人问题 `MC.pl`代码的介绍，你将会了解`lab2`目录结构，以及程序运行的方法。



## 运行环境

- 操作系统：`Ubuntu20.04 LST`
- 解释器：`SWI-Prolog version 7.6.4 for amd64`

如果您没有安装`SWI-Prolog`，可以使用如下命令安装它

```bash
$ sudo apt install swi-prolog
$ prolog --version
SWI-Prolog version 7.6.4 for amd64
```



## 文件目录

```
$ tree
.
├── MC.pl
└── README.md

0 directories, 2 files
```

- `MC.pl`：测试数据集
- `README.md`：说明文档



## 如何运行？

## 如何运行？

- 键入命令prolog加载`MC.pl`

```bash
$ prolog MC.pl 
?- find.
?- halt.
```

- 执行结果：

```bash
传教士和野人的数量为:
Missionaries:   =3
Cannibals:      =3

执行推理：

[3,3,left,0,0] --> [1,3,right,2,0]
[1,3,right,2,0] --> [2,3,left,1,0]
[2,3,left,1,0] --> [0,3,right,3,0]
[0,3,right,3,0] --> [1,3,left,2,0]
[1,3,left,2,0] --> [1,1,right,2,2]
[1,1,right,2,2] --> [2,2,left,1,1]
[2,2,left,1,1] --> [2,0,right,1,3]
[2,0,right,1,3] --> [3,0,left,0,3]
[3,0,left,0,3] --> [1,0,right,2,3]
[1,0,right,2,3] --> [1,1,left,2,2]
[1,1,left,2,2] --> [0,0,right,3,3]

推理结束
路径代价为:11
true .
```

