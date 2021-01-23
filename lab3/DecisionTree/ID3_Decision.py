import numpy as np
import pandas as pd
from utils.plotDecisionTree import *
from sklearn.metrics import accuracy_score
import time

#计算经验熵
def calcEntropy(dataSet):
    mD = len(dataSet)                               # mD表示数据集的数据向量个数
    dataLabelList = [x[-1] for x in dataSet]        # 数据集最后一列 标签
    dataLabelSet = set(dataLabelList)               # 转化为标签集合，集合不重复，所以转化
    ent = 0
    for label in dataLabelSet:                      # 对于集合中的每一个标签
        mDv = dataLabelList.count(label)            # 统计它出现的次数
        prop = float(mDv) / mD                      # 计算频率
        ent = ent - prop * np.math.log(prop, 2)     # 计算条件熵，见算法预备知识
    return ent

#计算条件熵
def calcCondEntropy(dataSet,featureSet,i):
    mD = len(dataSet)  
    ent=0.0
    for feature in featureSet:
        # 拆分数据集，去除第i行数据特征
        splitedDataSet = splitDataSet(dataSet, i, feature)  
        mDv = len(splitedDataSet)
        # H(D) - H(D|A) 计算信息增益
        ent = ent + float(mDv) / mD * calcEntropy(splitedDataSet)
    return ent


# 拆分数据集
# index   要拆分的特征的下标
# feature 要拆分的特征
# 返回值   dataSet中index所在特征为feature，且去掉index一列的集合
def splitDataSet(dataSet, index, feature):
    splitedDataSet = [] 
    for data in dataSet:
        if(data[index] == feature):                 # 将数据集拆分
            sliceTmp = data[:index]                 # 取[0,index)
            sliceTmp.extend(data[index + 1:])       # 扩展(index,len]
            splitedDataSet.append(sliceTmp)
    return splitedDataSet


# 根据信息增益 - 选择最好的特征
# 返回值 - 最好的特征的下标
def chooseBestFeature(dataSet):                     
    entD = calcEntropy(dataSet)                     # 计算经验熵
    mD = len(dataSet)               
    featureNumber = len(dataSet[0]) - 1
    maxGain = -100                                  # 最大增益
    maxIndex = -1                                   # 最大增益的下标
    for i in range(featureNumber):                      
        featureI = [x[i] for x in dataSet]          # 数据集合中的第i列特征
        featureSet = set(featureI)                  # 特征集合
        
        Gain = entD - calcCondEntropy(dataSet,featureSet,i) # 计算信息增益

        if(maxIndex == -1):
            maxGain = Gain
            maxIndex = i
        elif(maxGain < Gain):                       # 记录最大的信息增益和下标
            maxGain = Gain
            maxIndex = i
    return maxIndex                                 # 返回下标    


# 寻找最多的，作为标签
def mainLabel(labelList):
    labelRec = labelList[0]
    maxLabelCount = -1
    labelSet = set(labelList)
    for label in labelSet:
        if(labelList.count(label) > maxLabelCount):
            maxLabelCount = labelList.count(label)
            labelRec = label
    return labelRec

# 生成决策树，注意，是列表的形式存储
# dataSet：数据集, featureNames：数据属性类别, featureNamesSet：属性类别集合, labelListParent：父节点标签列表
def createFullDecisionTree(dataSet, featureNames, featureNamesSet, labelListParent):
    labelList = [x[-1] for x in dataSet]
    if(len(dataSet) == 0):                                  # 如果数据集为空，返回父节点标签列表的主要标签
        return mainLabel(labelListParent)
    elif(len(dataSet[0]) == 1):                             # 没有可划分的属性，选出最多的label作为该数据集的标签
        return mainLabel(labelList)                         
    elif(labelList.count(labelList[0]) == len(labelList)):  # 全部都属于同一个Label，返回labList[0]
        return labelList[0]

    # 不满足上面的边界情况则需要创建新的分支节点
    bestFeatureIndex = chooseBestFeature(dataSet)           # 根据信息增益，选择数据集中最好的特征下标
    bestFeatureName = featureNames.pop(bestFeatureIndex)    # 取出属性类别
    myTree = {bestFeatureName: {}}                          # 新建节点，一个字典
    featureList = featureNamesSet.pop(bestFeatureIndex)     # 取出最佳属性的类别
    featureSet = set(featureList)                           # 剔除属性类别集合
    for feature in featureSet:                              # 遍历最佳属性所有取值
        featureNamesNext = featureNames[:]                  
        featureNamesSetNext = featureNamesSet[:][:]
        splitedDataSet = splitDataSet(dataSet, bestFeatureIndex, feature)   # 剔除最佳特征
        # 递归地生成新的节点
        # featureNames：数据属性类别, featureNamesSet：属性类别集合, labelListParent：父节点标签列表
        # 一个二叉树
        myTree[bestFeatureName][feature] = createFullDecisionTree(splitedDataSet, featureNamesNext, featureNamesSetNext, labelList)
    return myTree


# 读取数据集
def readDataSet(path):
    ifile = open(path)
    #表头
    featureName = ifile.readline()              
    featureName = featureName.rstrip("\n")
    #类别,属性
    featureNames = (featureName.split(' ')[0]).split(',')
    #读取文件
    lines = ifile.readlines()
    #数据集
    dataSet = []
    for line in lines:
        tmp = line.split('\n')[0]
        tmp = tmp.split(',')
        dataSet.append(tmp)
    #获取标签
    labelList = [x[-1] for x in dataSet]
    #获取featureNamesSet
    featureNamesSet = []
    for i in range(len(dataSet[0]) - 1):
        col = [x[i] for x in dataSet]
        colSet = set(col)
        featureNamesSet.append(list(colSet))
    #返回 数据集，属性名，所有属性的取值集合，以及标签列表
    return dataSet, featureNames, featureNamesSet,labelList

def tree_predict(tree, data):
  #print(data)
  feature = list(tree.keys())[0]    #取树第一个结点的键（特征）
  #print(feature)
  label = data[feature]             #该特征下的属性
  next_tree = tree[feature][label]  #取下一个结点树
  if type(next_tree) == str:        #如果是个字符串，说明已经到达叶节点返回分类结果
    return next_tree
  else:                             # 否则继续如上处理
    return tree_predict(next_tree, data)

def main():
    #获取训练集,所有属性名称，每个属性的类别，所有标签(最后一列)    
    dataTrain, featureNames, featureNamesSet,labelList = readDataSet("../data/test.txt")

    #获取测试集
    train= pd.read_csv("../data/test.txt")
    test = pd.read_csv("../data/predict.txt")
    print("train:\n",train[:10])
    print("test:\n",test[:10])

    #生成决策树
    t0 = time.time()
    tree=createFullDecisionTree(dataTrain, featureNames,featureNamesSet,labelList)
    t1 = time.time()
    print("ID3算法生成决策树的时间开销：",(t1 - t0)*(10**3),"ms")
    # createPlot(tree,"fig/ID3.png")

    predictTrain = train.apply(lambda x: tree_predict(tree, x), axis=1)
    label_list = train.iloc[:, -1]
    score = accuracy_score(label_list, predictTrain)
    print('训练补全分支准确率为：' + repr(score * 100) + '%')

    #预测
    y_predict = test.apply(lambda x: tree_predict(tree, x), axis=1)
    label_list = test.iloc[:, -1]
    score = accuracy_score(label_list, y_predict)
    print('测试集补全分支准确率为：' + repr(score * 100) + '%')

if __name__ == "__main__":
    main()