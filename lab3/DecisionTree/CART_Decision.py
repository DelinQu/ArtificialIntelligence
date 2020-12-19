import numpy as np
import pandas as pd
import time
from sklearn.metrics import accuracy_score
from utils.plotDecisionTree import *

# 拆分数据集
# index   要拆分的特征的下标
# feature 要拆分的特征
# 返回值   dataSet中index所在特征为feature，且去掉index一列的集合
def splitDataSet(dataSet, index, feature):
    splitedDataSet = []
    mD = len(dataSet)   
    for data in dataSet:
        if(data[index] == feature):                 # 将数据集拆分
            sliceTmp = data[:index]                 # 取[0,index)
            sliceTmp.extend(data[index + 1:])       # 扩展(index,len]
            splitedDataSet.append(sliceTmp)
    return splitedDataSet

# 计算基尼值
def calcGini(splitedDataSet):
    gini_index = 1
    y = list(splitedDataSet[-1])                    # 标签列表
    for unique_val in set(y):                       # 对于标签列表中的每一个value
        p = y.count(unique_val) / len(y)            # 计算它出现的概率
        gini_index -= p**2                          # 计算gini-=p^2
    return gini_index

# 根据信息增益比，选择最佳的特征，并且返回最佳特征的下标
def chooseBestFeature_CART(dataSet):                           
    featureNumber = len(dataSet[0]) - 1
    minGini = 1000000.0                             # 最小基尼指数
    minIndex = -1                                   # 下标
    Gini=0
    for i in range(featureNumber):                      
        featureI = [x[i] for x in dataSet]          # 数据集合中的第i列特征
        featureSet = set(featureI)                  # 特征集合
        # 计算基尼指数
        for feature in featureSet:                  # 遍历某一特征的所有属性
            splitedDataSet = splitDataSet(dataSet, i, feature) # 特征将其划分为Di
            count = len(splitedDataSet)                   
            # 计算Di / D * Gini(Di)
            Gini += (count / len(dataSet[0]))*calcGini(splitedDataSet)
        if(minIndex == -1):
            minGini = Gini
            minIndex = i
        elif(minGini < Gini):                       # 记录最小基尼和下标
            minGini = Gini
            minIndex = i
    return minIndex                                 # 返回下标    


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
    bestFeatureIndex = chooseBestFeature_CART(dataSet)      # 根据信息增益，选择数据集中最好的特征下标
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
    print("CART算法生成决策树的时间开销：",(t1 - t0)*(10**6),"us")
    createPlot(tree,"fig/CART.png")

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