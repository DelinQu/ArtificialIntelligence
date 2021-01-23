# 使用sklearn完成4种基本的分类算法：朴素贝叶斯算法、决策树算法、人工神经网络、支持向量机算法
from sklearn import svm,tree
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
# 导入库与数据集：训练集&测试集
import numpy as np
import time

# unacc,acc,good,vgood
def iris_type1(s): #class value
    Class_Values = {b'unacc': 0, b'acc': 1, b'good': 2, b'vgood': 3}
    return Class_Values[s]

def iris_type2(s): #buying
    buying = {b'vhigh': 0, b'high': 1, b'med': 2, b'low': 3}
    return buying[s]

def iris_type3(s): # maint
    maint =  {b'vhigh': 0, b'high': 1, b'med': 2, b'low': 3}
    return maint[s]

def iris_type4(s): # doors
    doors = {b'2':0, b'3':1, b'4':2, b'5more':3}
    return doors[s]

def iris_type5(s):# person
    persons = {b'2':0, b'4':1, b'more':2}
    return persons[s]

def iris_type6(s): # luggage_boot
    lug_boot = {b'small':0, b'med':1, b'big':2}
    return lug_boot[s]

def iris_type7(s):  # safty
    safety = {b'low':0, b'med':1, b'high':2}
    return safety[s]



train = np.loadtxt('./sklearns/test.txt',dtype=int,delimiter=',',converters={0: iris_type2, 1: iris_type3, 2: iris_type4, 3: iris_type5, 4: iris_type6, 5: iris_type7, 6: iris_type1})
test = np.loadtxt('./sklearns/predict.txt',skiprows=1,dtype=int,delimiter=',',converters={0: iris_type2, 1: iris_type3, 2: iris_type4,3: iris_type5, 4: iris_type6, 5: iris_type7,6: iris_type1})
train_x, train_y = np.split(train, (6,), axis=1) # 划分特征 和 标签
test_x, test_y = np.split(test, (6,), axis=1) # 划分属性 和 value

# 1：水平分割 0：垂直分割
if __name__ == '__main__':

    print("train:\n",train)
    print("test:\n",train)

    # SVM C是惩罚系数
    print("-------SVM线性核-------")
    clf = svm.SVC(C=0.8, kernel='linear', decision_function_shape='ovr')

    t0 = time.time()
    clf.fit(train_x, train_y.ravel())
    t1 = time.time()
    print("SVM时间开销：",(t1 - t0)*(10**3),"ms")

    print("训练准确率:\n",clf.score(train_x, train_y))
    print("测试准确率:\n",clf.score(test_x, test_y))

    # 高斯核
    print("-------SVM高斯核-------")
    clf = svm.SVC(C=0.85, kernel='rbf', decision_function_shape='ovr')
    t0 = time.time()
    clf.fit(train_x, train_y.ravel())
    t1 = time.time()
    print("SVM时间开销：",(t1 - t0)*(10**3),"ms")
    print("训练准确率:\n",clf.score(train_x, train_y))
    print("测试准确率:\n",clf.score(test_x, test_y))