import numpy as np
from numpy.core.fromnumeric import shape
import pandas as pd
import time

# 加载数据，返回1位编码数据特征矩阵和label矩阵
def load_data(path):
    data = pd.read_csv(path)
    print("data_before encoding:\n",data)
    # 一位编码，使得可以在神经网络中计算
    data = hash_encode(data)
    print("data_after_encoding:\n",data)
    data_mat = data.values
    col_num = data_mat.shape[1]
    return data_mat[:, 0:col_num-1], data_mat[:, col_num-1]

# 进行一位编码
def hash_encode(df):
    # 使用factorize编码标签
    df['Class_Values'] = pd.factorize(df['Class_Values'])[0]
    # 编码数据特征
    df = pd.get_dummies(df)
    # 将label插入结尾处
    cols = list(df)
    cols.pop(cols.index('Class_Values'))
    cols.append('Class_Values')
    df = df.loc[:, cols]
    return df

# 神经网络
class NeuralNetwork:
    # 初始化
    def __init__(self, learning_rate, layers=1, iter_nums=5000):
        self.lr = learning_rate     # 学习率
        self.layers = layers        # 神经隐藏层数    
        self.hide_node = 3          # 隐藏层节点数
        self.iters = iter_nums      # 训练次数
        self.weight = []            # 权值         
        self.bias = []              # 偏移量
    
    # 激活函数sigmod
    def __active_fun(self, x):
        y = 1 / (1 + np.exp(-x))
        return y
    
    # 初始化w， b
    def __init_para(self, x_train):
        rows, cols = x_train.shape
        w_1 = np.random.randn(cols, self.hide_node)
        # 返回用0填充的数组,1*hide_node规模
        b_1 = np.zeros((1, self.hide_node))
        w_2 = np.random.randn(self.hide_node, 1)
        b_2 = np.zeros((1, 1))
        return w_1, w_2, b_1, b_2

    # 计算error
    def __mean_square_error(self, predict_y, y):
        differ = predict_y - y
        error = 0.5*sum(differ**2)
        return error

    # 训练拟合
    def fit(self, x_train, y_train):
        self.hide_node = int(np.round(np.sqrt(x_train.shape[0])))               # 隐藏层节点数
        # initialize the network parameter
        w_1, w_2, b_1, b_2 = self.__init_para(x_train)                          # 初始化权值w，阈值b
        for i in range(self.iters):                                             # 迭代iters次
            accum_error = 0
            for s in range(x_train.shape[0]):                                   # 遍历每一个数据输入
                hidein_1 = np.dot(x_train[s, :], w_1) + b_1                     # 计算隐藏层1 y1 = w1*x +b1
                hideout_1 = self.__active_fun(hidein_1)                         # 激活
                hidein_2 = np.dot(hideout_1, w_2) + b_2                         # 计算隐藏层2 y2 = w2*x +b2
                hideout_2 = self.__active_fun(hidein_2)                         # 激活    
                predict_y = hideout_2                                           
                accum_error += self.__mean_square_error(predict_y, y_train[s])  # 计算误差error，第二层
            if accum_error <= 0.001:                                            # 如果误差小于0.001,我们认为这个权值可以接受
                break
            else:  # update the parameter                                       # 否则更新，开始反向传播BP
                for s in range(x_train.shape[0]):                               
                    in_nums = x_train.shape[1]
                    # layer 1
                    hidein_1 = np.dot(x_train[s, :], w_1) + b_1                 # 重新计算
                    hideout_1 = self.__active_fun(hidein_1)
                    # layer 2
                    hidein_2 = np.dot(hideout_1, w_2) + b_2
                    hideout_2 = self.__active_fun(hidein_2)
                    predict_y = hideout_2

                    g = predict_y*(1 - predict_y)*(y_train[s] - predict_y)      # 计算权值增量g
                    e = g*w_2.T*(hideout_1*(1 - hideout_1))                     # 计算误差
                    w_2 = w_2 + self.lr*hideout_1.T*g                           # 更新w2
                    b_2 = b_2 - self.lr*g                                       # 更新b2
                    w_1 = w_1 + self.lr*x_train[s, :].reshape(in_nums, 1)*e     # 更新w1
                    b_1 = b_1 - self.lr*e                                       # 更新b1
        self.weight.append(w_1)
        self.weight.append(w_2)
        self.bias.append(b_1)
        self.bias.append(b_2)

    # 结果预测
    def predict(self, x_train,y_train):
        # 准确率
        accRate=0
        # dot向量点积和矩阵乘法
        print(shape(x_train),shape(self.weight[0]),self.bias[0])
        hidein_1 = np.dot(x_train, self.weight[0]) + self.bias[0]
        hideout_1 = self.__active_fun(hidein_1)

        hidein_2 = np.dot(hideout_1, self.weight[1]) + self.bias[1]
        hideout_2 = self.__active_fun(hidein_2)
        predict_y = hideout_2                                                   # 计算第二层的输出
        cur=[]
        for e in predict_y:                                                     # 对于输出向量，对区间进行划分[0,0.5),[0.5,1.5),[1.5,2.5),[2.5,--)
            if e>2.5:
                cur.append(3)
            elif e>1.5:
                cur.append(2)
            elif e>0.5:
                cur.append(1)
            else:
                cur.append(0)
        predict_y = cur
        for i in range(len(predict_y)):                                         
            if(predict_y[i]==y_train[i]):
                accRate=accRate+1
        # 返回准确率
        return accRate/len(predict_y)

if __name__ == "__main__":
    # 数据总集
    data_x, data_y = load_data("./data/dataset.txt")
    print("shapX:",shape(data_x))
    # 训练集
    train_x,train_y=data_x[:1350],data_y[:1350]
    print("train_x:",shape(train_x),"train_y",shape(train_y))
    # 测试集
    test_x,test_y=data_x[1350:],data_y[1350:]
    print("test_x:",shape(test_x),"test_y",shape(test_y))

    # 学习率为0.1
    model = NeuralNetwork(0.1)

    t0 = time.time()
    model.fit(train_x,train_y)
    t1 = time.time()
    print("BP算法的时间开销：",(t1 - t0),"s")
    
    # 预测训练集准确率
    print("训练准确率为: ",model.predict(train_x,train_y))
    print("测试准确率为: ",model.predict(test_x,test_y))
