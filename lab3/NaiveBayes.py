#encoding:utf-8
import pandas as pd
import numpy  as np
import time

class NaiveBayes:
    def __init__(self):
        self.model = {}         #key 为类别名 val 为字典PClass表示该类的该类，PFeature:{}对应对于各个特征的概率

    def calEntropy(self, y):    # 计算熵
        valRate = y.value_counts().apply(lambda x : x / y.size) # 频次汇总 得到各个特征对应的概率
        valEntropy = np.inner(valRate, np.log2(valRate)) * -1
        return valEntropy

    def fit(self, xTrain, yTrain = pd.Series()):
        if not yTrain.empty:    #如果不传，自动选择最后一列作为分类标签
            xTrain = pd.concat([xTrain, yTrain], axis=1)
        self.model = self.buildNaiveBayes(xTrain) 
        return self.model
        
    # 构建贝叶斯算法
    def buildNaiveBayes(self, xTrain): 
        yTrain = xTrain.iloc[:,-1]              # 标签labels
        yTrainCounts = yTrain.value_counts()    # 得到各个标签的个数
        # 使用拉普拉斯平滑 P(ci) = Dc+1 / D + N, Dc为该类别的频数，N表示所有类别的可能数。 
        yTrainCounts = yTrainCounts.apply(lambda x : (x + 1) / (yTrain.size + yTrainCounts.size)) 
        retModel = {}

        # 遍历标签集合中的每一项，用字典的数据结构保存，将会作为字典树返回
        for nameClass, val in yTrainCounts.items():
            retModel[nameClass] = {'PClass': val, 'PFeature':{}}

        # 训练集所有特征
        propNamesAll = xTrain.columns[:-1]
        allPropByFeature = {}

        # 遍历标特征集合，计算每个特征的数量，这个关系字典allPropByFeature十分重要，在后续将会用到
        for nameFeature in propNamesAll:
            allPropByFeature[nameFeature] = list(xTrain[nameFeature].value_counts().index)
        
        # 使用groupby函数根据标签进行分组，并且遍历它
        for nameClass, group in xTrain.groupby(xTrain.columns[-1]):
            # 遍历训练集中的所有特征a
            for nameFeature in propNamesAll:
                eachClassPFeature = {}                                                          
                propDatas = group[nameFeature]                                                  # 当前特征取值
                propClassSummary = propDatas.value_counts()                                     # 频次汇总 得到各个特征对应的频数
                for propName in allPropByFeature[nameFeature]:                                  # 遍历每个特征的属性ai
                    if not propClassSummary.get(propName):                                      # 如果有属性没有，那么自动补0
                        propClassSummary[propName] = 0                                          
                Ni = len(allPropByFeature[nameFeature])                                         # Ni表示所有特征出现的可能数
                #使用拉普拉斯平滑 P(ci) = Dc+1 / D + N, Dc为该类别的频数，N表示所有类别的可能数。
                propClassSummary = propClassSummary.apply(lambda x : (x + 1) / (propDatas.size + Ni))
                # 我们现在已经统计好了每个ai的概率，以字典的结构保存在propClassSummary中，我们要将其映射到eachClassPFeature上，然后一并返回
                for nameFeatureProp, valP in propClassSummary.items():
                    eachClassPFeature[nameFeatureProp] = valP

                # 保存每个特征的概率ai，返回
                retModel[nameClass]['PFeature'][nameFeature] = eachClassPFeature
        return retModel
    
    # 预测分类，取最大值
    def predictBySeries(self, data):
        curMaxRate = None                                                                       # 概率最大值
        curClassSelect = None                                                                   # 最大概率对应的类
        for nameClass, infoModel in self.model.items():                                         # 遍历朴素贝叶斯结构中的每一个类
            rate = 0                                                                            # 当前标签类nameClass的概率
            # 为防止由于某些特征属性的值P(Xi|Ci)可能很小，多个特征的p值连乘后可能被约等于0
            # 取log然后变乘法为加法，避免连乘问题。 
            rate += np.log(infoModel['PClass'])                                                 # 求和条件概率
            PFeature = infoModel['PFeature']                                                    # 取出当前特征
            # 遍历数据集中的每一个特征向量
            for nameFeature, val in data.items():
                propsRate = PFeature.get(nameFeature)                                           # 获取当前特征发生的概率a
                if not propsRate:                                                               # 为0则直接跳过
                    continue
                #使用log加法避免很小的小数连续乘，接近零
                rate += np.log(propsRate.get(val, 0))                                           # 计算当前标签类nameClass的概率
            if curMaxRate == None or rate > curMaxRate:                                         # 迭代，取最大值
                curMaxRate = rate
                curClassSelect = nameClass

        return curClassSelect                                                                   # 返回被选择的类

    # 与的结果    
    def predict(self, data):
        if isinstance(data, pd.Series):
            return self.predictBySeries(data)
        return data.apply(lambda d: self.predictBySeries(d), axis=1)

if __name__ == "__main__":
    dataTrain = pd.read_csv("./data/test.txt", encoding = "gbk")
    dataPredict = pd.read_csv("./data/predict.txt", encoding = "gbk")
    print(dataTrain)

    naiveBayes = NaiveBayes()
    t0 = time.time()
    treeData = naiveBayes.fit(dataTrain)
    t1 = time.time()
    print("NaiveBayes算法的时间开销：",(t1 - t0)*(10**3),"ms")


    # 训练准确率
    print("-------------------------------------------\n训练准确率")
    pd1 = pd.DataFrame({'预测值':naiveBayes.predict(dataTrain), '正取值':dataTrain.iloc[:,-1]})
    print(pd1)
    print('正确率:%f%%'%(pd1[pd1['预测值'] == pd1['正取值']].shape[0] * 100.0 / pd1.shape[0]))

    print("-------------------------------------------\n测试准确率")
    # 测试准确率
    pd2 = pd.DataFrame({'预测值':naiveBayes.predict(dataPredict), '正取值':dataPredict.iloc[:,-1]})
    print(pd2)
    print('正确率:%f%%'%(pd2[pd2['预测值'] == pd2['正取值']].shape[0] * 100.0 / pd2.shape[0]))
