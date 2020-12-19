# 使用sklearn完成决策树算法
from sklearn import tree
# 导入库与数据集：训练集&测试集
import numpy as np
import graphviz
import time

def iris_type1(s): # 销量
    sales = {b'low': 0, b'high': 1}
    return sales[s]

def iris_type2(s): # 天气
    weather = {b'bad': 0, b'good': 1}
    return weather[s]

def iris_type3(s): # 是否周末
    isWeekend =  {b'false': 0, b'true': 1}
    return isWeekend[s]

def iris_type4(s): # 是否有促销
    promotion = {b'false':0,b'true':1}
    return promotion[s]


# 测试和训练
train = np.loadtxt('./data/ex3dataEn.csv',dtype=int,delimiter=',',converters={0: iris_type2, 1: iris_type3, 2: iris_type4, 3: iris_type1})
test = np.loadtxt('./data/ex3dataEn.csv',dtype=int,delimiter=',',converters={0: iris_type2, 1: iris_type3, 2: iris_type4,3: iris_type1})
train_x, train_y = np.split(train, (3,), axis=1)    # 划分特征 和 标签
test_x, test_y = np.split(test, (3,), axis=1)       # 划分属性 和 value

# 1：水平分割 0：垂直分割
if __name__ == '__main__':
    # print("train:\n",train)
    # print("test:\n",test)
    # 决策树(Decision Tree）
    # 使用信息熵作为划分标准，对决策树进行训练
    clf = tree.DecisionTreeClassifier(criterion='entropy')
    t0=time.time()
    clf.fit(train_x, train_y.ravel())
    t1=time.time()
    print("sklearn生成决策树的时间开销：",(t1 - t0)*(10**6),"us")

    print("------决策树 by sklearn------")
    print("训练准确率：",clf.score(train_x, train_y))
    print(clf.score(test_x, test_y))

    dot_data = tree.export_graphviz(clf, out_file='./fig/tree.dot',  
                                feature_names=['weather', 'isWeekend', 'promotion'],
                                class_names=['low','high'],  
                                filled=True, rounded=True, 
                                special_characters=True)
    with open("./fig/tree.dot") as f:
        dot_graph = f.read()

    dot=graphviz.Source(dot_graph,filename="./fig/sklearn")
    
    dot.view()