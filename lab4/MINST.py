#导入需要的包
import torch 
from torch import nn
import matplotlib.pyplot as plt
import os
from torchvision import datasets, transforms,utils
import torch.nn.functional as F
import torch.optim as optim

PATH='./fig/'

# 从二维数组生成一张图片
def showOneImg(train_data):
    oneimg,label = train_data[0]
    oneimg = oneimg.numpy().transpose(1,2,0) 
    std = [0.5]
    mean = [0.5]
    oneimg = oneimg * std + mean
    oneimg.resize(28,28)
    plt.imshow(oneimg)
    plt.savefig(PATH+"one.png")
    plt.show()

# 输出一个batch的图片和标签
def showBatchImg(train_loader):
    images, lables = next(iter(train_loader))
    img = utils.make_grid(images)
    # transpose 转置函数(x=0,y=1,z=2),新的x是原来的y轴大小，新的y是原来的z轴大小，新的z是原来的x大小
    #相当于把x=1这个一道最后面去。
    img = img.numpy().transpose(1,2,0) 
    std = [0.5]
    mean = [0.5]
    img = img * std + mean
    for i in range(64):
        print(lables[i], end=" ")
        i += 1
        if i%8 == 0:
            print(end='\n')
    plt.imshow(img)
    plt.savefig(PATH+"Batch.png")
    plt.show()

class Model(nn.Module):
    def __init__(self) -> None:
        super(Model,self).__init__()
        # 卷积层1
        self.conv1 = nn.Conv2d(1,32,kernel_size=3,stride=1,padding=1)
        self.pool = nn.MaxPool2d(2,2)
        # 卷积层2
        self.conv2 = nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1)
        self.fc1 = nn.Linear(64*7*7,1024)   #两个池化，所以是7*7而不是14*14
        self.fc2 = nn.Linear(1024,512)
        self.fc3 = nn.Linear(512,10)

    # 前向传播的过程
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7* 7)           #将数据平整为一维的 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))   
        x = self.fc3(x)  
        # x = F.log_softmax(x,dim=1) NLLLoss()才需要，交叉熵不需要
        return x


# 定义神经卷积网络 CNN
class CNN():
    # 训练过程
    train_accs = []             # 训练准确率
    train_loss = []             # 训练损失率
    test_accs = []              # 测试准确率
    PATH = './mnist_net.pth'    # 保存路径

    def __init__(self,lr=0.001, epochs=15,PATH="./mnist_net.pth"):
        self.model = Model()
        self.criterion = nn.CrossEntropyLoss()      # 交叉熵
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9) # 随机梯度下降优化器
        #也可以选择Adam优化方法
        # self.optimizer = torch.optim.Adam(net.parameters(),lr=1e-2)
        # self.dp = nn.Dropout(p=0.5)
        self.epochs = epochs
        self.PATH=PATH

    # 训练拟合
    def fit(self,train_loader):
        net = self.model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = net.to(device)
        for epoch in range(self.epochs):
            running_loss = 0.0
            for i,data in enumerate(train_loader,0):#0是下标起始位置默认为0
                # data 的格式[[inputs, labels]]       
                # inputs,labels = data
                inputs,labels = data[0].to(device), data[1].to(device)
                #初始为0，清除上个batch的梯度信息
                self.optimizer.zero_grad()
                
                #前向+后向+优化     
                outputs = net(inputs)
                loss = self.criterion(outputs,labels)
                loss.backward()
                self.optimizer.step()
                
                # loss 的输出，每个一百个batch输出，平均的loss
                running_loss += loss.item()
                if i%100 == 99:
                    print('[%d,%5d] loss :%.3f' %
                        (epoch+1,i+1,running_loss/100))
                    running_loss = 0.0
                self.train_loss.append(loss.item())
                
                # 训练曲线的绘制 一个batch中的准确率
                correct = 0
                total = 0
                _, predicted = torch.max(outputs.data, 1)
                total = labels.size(0)# labels 的长度
                correct = (predicted == labels).sum().item() # 预测正确的数目
                self.train_accs.append(100*correct/total)     
        # 训练完成
        print('Finished Training')
        # 保存训练结果
        torch.save(net.state_dict(), self.PATH)


    # 绘制训练过程图像
    def draw_train_process(self,title="training",label_cost="training loss",lable_acc="training acc"):
        iters=range(len(self.train_accs))
        costs=self.train_loss
        accs=self.train_accs
        plt.title(title, fontsize=24)
        plt.xlabel("iter", fontsize=20)
        plt.ylabel("acc(\%)", fontsize=20)
        plt.plot(iters, costs,color='red',label=label_cost) 
        plt.plot(iters, accs,color='green',label=lable_acc) 
        plt.legend()
        plt.grid()
        plt.savefig(PATH+"process.png")
        plt.show()

    # 检验一个batch的分类情况
    def predOneBatch(self,test_loader):
        dataiter = iter(test_loader)
        images, labels = dataiter.next()
        # print images
        test_img = utils.make_grid(images)
        test_img = test_img.numpy().transpose(1,2,0)
        std = [0.5,0.5,0.5]
        mean =  [0.5,0.5,0.5]
        test_img = test_img*std+0.5
        plt.imshow(test_img)
        plt.savefig(PATH+"preBatch.png")
        plt.show()
        print('GroundTruth: ', ' '.join('%d' % labels[j] for j in range(64)))

        self.model.load_state_dict(torch.load(self.PATH))
        test_out = self.model(images)
        print(test_out)
        _, predicted = torch.max(test_out, dim=1)
        print('Predicted: ', ' '.join('%d' % predicted[j] for j in range(64)))


    # 预测测试集
    def pred(self,test_loader):
        dataiter = iter(test_loader)
        images, labels = dataiter.next()
        self.model.load_state_dict(torch.load(self.PATH))
        test_out = self.model(images)

        # 预测结果，取max
        print(test_out)
        _, predicted = torch.max(test_out, dim=1)

        print('Predicted: ', ' '.join('%d' % predicted[j] for j in range(64)))

        # 测试集上面整体的准确率
        correct = 0
        total = 0
        with torch.no_grad():           # 进行评测的时候网络不更新梯度
            for data in test_loader:
                images, labels = data
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0) # labels 的长度
                correct += (predicted == labels).sum().item() # 预测正确的数目
        print('Accuracy of the network on the  test images: %f %%' % (100. * correct / total))

    # 预测10个类别的准确率
    def predLabels(self,test_loader):
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        self.model.load_state_dict(torch.load(self.PATH))

        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels)
                # print(predicted == labels)
                for i in range(10):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        for i in range(10):
            print('Accuracy of %d : %4f %%' % (
                i, 100 * class_correct[i] / class_total[i]))


if __name__ == "__main__":
    # 导入数据
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.5],std=[0.5])])
    train_data = datasets.MNIST(root = "./data/",transform=transform,train = True,download = True)
    test_data = datasets.MNIST(root="./data/",transform = transform,train = False)
    # 加载数据
    train_loader = torch.utils.data.DataLoader(train_data,batch_size=64,shuffle=True,num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_data,batch_size=64,shuffle=True,num_workers=2)

    # 显示一张图片
    showOneImg(train_data)

    # 显示一批图片
    showBatchImg(train_loader)

    dataiter = iter(test_loader)
    images, labels = dataiter.next()
    print(labels)

    cnn=CNN()

    # 训练
    cnn.fit(train_loader)

    # 绘制训练过程
    cnn.draw_train_process()

    # 检验一个batch的分类情况
    cnn.predOneBatch(test_loader)

    # 预测测试集
    cnn.pred(test_loader)

    # 预测10个类别的准确率
    cnn.predLabels(test_loader)