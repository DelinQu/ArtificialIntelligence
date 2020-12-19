# 使用sklearn完成4种基本的分类算法：朴素贝叶斯算法、决策树算法、人工神经网络、支持向量机算法
from sklearn import svm,tree
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
# 导入库与数据集：训练集&测试集
import numpy as np

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



train = np.loadtxt('test.txt',dtype=int,delimiter=',',converters={0: iris_type2, 1: iris_type3, 2: iris_type4, 3: iris_type5, 4: iris_type6, 5: iris_type7, 6: iris_type1})
test = np.loadtxt('predict.txt',skiprows=1,dtype=int,delimiter=',',converters={0: iris_type2, 1: iris_type3, 2: iris_type4,3: iris_type5, 4: iris_type6, 5: iris_type7,6: iris_type1})
train_x, train_y = np.split(train, (6,), axis=1) # 划分特征 和 标签
test_x, test_y = np.split(test, (6,), axis=1) # 划分属性 和 value

# 1：水平分割 0：垂直分割
if __name__ == '__main__':

    # GaussianNB(高斯朴素贝叶斯)
    clf = GaussianNB()  # fit:监督学习算法，训练模型
    clf.fit(train_x, train_y.ravel())  # ravel:多维数组转换为一维数组
    print("----高斯朴素贝叶斯----")
    print(clf.score(train_x, train_y))
    print(clf.score(test_x, test_y))

    #决策树(Decision Tree）
    #使用信息熵作为划分标准，对决策树进行训练
    clf = tree.DecisionTreeClassifier(criterion='entropy')
    clf.fit(train_x, train_y.ravel())
    print("------决策树------")
    print(clf.score(train_x, train_y))
    print(clf.score(test_x, test_y))

    #人工神经网络
    clf = MLPClassifier(hidden_layer_sizes=(12, 9), activation='logistic', solver='lbfgs', alpha=1e-3, random_state=1, max_iter=1500)
    # 第一层12个神经元 第二层9个神经元
    # f(x)=1/(1+exp(-x));
    # lbfgs:准牛顿方法的优化器
    # alpha:学习率：控制权重的更新比率
    # random_state:重排数据训练集

    clf.fit(train_x, train_y.ravel())
    print("----人工神经网络----")
    print(clf.score(train_x, train_y))
    print(clf.score(test_x, test_y))

    # SVM C是惩罚系数
    print("-------SVM线性核-------")
    clf = svm.SVC(C=0.8, kernel='linear', decision_function_shape='ovr')
    clf.fit(train_x, train_y.ravel())
    print("训练准确率:\n",clf.score(train_x, train_y))
    print("测试准确率:\n",clf.score(test_x, test_y))

    # 高斯核
    print("-------SVM高斯核-------")
    clf = svm.SVC(C=0.85, kernel='rbf', decision_function_shape='ovr')
    clf.fit(train_x, train_y.ravel())
    print("训练准确率:\n",clf.score(train_x, train_y))
    print("测试准确率:\n",clf.score(test_x, test_y))