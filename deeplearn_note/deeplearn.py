import os
import sys
sys.path.append(os.pardir)
from PIL import Image  # 导入图像显示模块
from common.util import im2col
from dataset.mnist import load_mnist
from collections import OrderedDict
import pickle
import numpy as np
import matplotlib.pyplot as plt
'''感知机
        x1w1+x2w2+(-theta)<=0输出0
        x1w1+x2w2+(-theta)> 0输出1
        b==theta
'''


def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    x = np.array([x1, x2])
    w = np.array([w1, w2])
    b = -theta
    temp = np.sum(x*w)+b
    if temp <= 0:
        return 0
    return 1


def OR(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.2
    x = np.array([x1, x2])
    w = np.array([w1, w2])
    b = -theta
    temp = np.sum(x*w)+b
    if temp <= 0:
        return 0
    return 1


def NAND(x1, x2):  # 与非门
    w1, w2, theta = -0.5, -0.5, -0.7
    x = np.array([x1, x2])
    w = np.array([w1, w2])
    b = -theta
    temp = np.sum(x*w)+b
    if temp <= 0:
        return 0
    return 1


def XOR(x1, x2):  # 异或门
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    return AND(s1, s2)
# 简化加入激活函数


def step_fun(x):  # 阶跃函数(支持数组)
    return np.array(x > 0, dtype=np.int)
    # 判断数组中的数是否大于0并将结果用astype转化为int型


def identifify_fun(x):
    return x  # 恒等函数


def sigmoid(x):  # s形函数
    return 1/(1+np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    '''可以将输出正规化为0~1之间的数据，其中值越大则转化后的概率也越大
    概率函数，y[0]的概率...y[1]的概率...
    yk=exp(xk)/sum-exp(xi)[1<=i<=n]
    防止指数函数运算溢出，将分子分母分别在指数内加一个系统上限数或者输入函数中的最大数'''
    if x.ndim == 2:
        x = x.T  # 求x的转置
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)  # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)


def relu_grad(x):
    grad = np.zeros(x)
    grad[x >= 0] = 1
    return grad


'''np.dot(矩阵点乘);np.shape(多维数组第0维元素个数，第1维元素个数...)=(3,2)三行两列'''
# 矩阵实现三层神经网络


def init_network_ex():
    '''第0层两个神经元，第1层三个神经元则有3个偏置权重b（分别代表第1层三个神经元的激活阈值）
    将神经网络第0层x数值用1*2矩阵X代替，第0层神经元流向第1层神经元——第1层权重用2*3矩阵W1代替
    偏置权重用1*3矩阵B1代替，第1层神经元的输入（即隐藏层（第1，2..层）的加权和）可以用A1代替
    隐藏层的加权和经过激活函数后的数值用Z1代替
    第2层第3层类推...
    注：举例为第0层（输入层）2个神经元，第1层3个，第2层2个，第3层（输出层）2个
    '''
    network = {}  # {}表示字典数据类型
    # 第0层到第1层的权重和偏置权重
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    # 第1层到第2层
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    # 第2层到输出层
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])
    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1)+b1  # a1,a2,a3=第1层的加权信号和+偏置权重
    z1 = sigmoid(a1)  # 加权和A经过激活函数后的信号记作Z
    a2 = np.dot(z1, W2)+b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3)+b3
    # y=identifify_fun(a3)#y=a3
    y = softmax(a3)
    # 输出层函数：回归问题=恒等函数，分类问题=sofmax函数
    return y


def done_network_ex():  # 输出神经网络输出层结果
    network = init_network_ex()
    x = np.array([1, 0.5])
    y = predict(network, x)
    print(y)
    print(np.sum(y))


# mnist的导入与处理
sys.path.append(os.pardir)  # 将父目录加入搜索路径


def img_show(img):
    '''可以打印一维数组存储的非标准化图像，若标准化则显示出来的图像是黑色
    img = np.asarray(image)图像转化为数组
    reimage=Image.fromarray(np.uint8(img))数组转化为图像
    img=img.reshape(28,28)将图片展开为原本的尺寸
    uint8：八位无符号数'''
    img = img.reshape(28, 28)  # 把图像的形状变成原来的尺寸
    pil_img = Image.fromarray(np.uint8(img))  # 实现从array到image的转换
    pil_img.show()


def get_data():
    '''(训练图像，训练标签)(测试图像，测试标签)
    flatten：true则将图像压缩为784大小一维数组，false则展开为1*28*28三维数组
    注意此处是压缩并标准化，正常显示的图像，输出的数据取得是测试图像和测试标签'''
    (x_train, t_train), (x_test, t_test) = load_mnist(
        flatten=True, normalize=True)
    return x_train, t_train, x_test, t_test


def init_mnist():
    '''利用pickle保存了network的数据直接导入即可
    这个文件中的输入神经元有784个，输出神经元10个，中间两层隐藏层，第1层50个第2层100个
    关于隐藏层的权重数量问题：w12为第1层的第2个神经元到第2层的第一个神经元的权重大小
    在矩阵中排列为[w11,w21,w31
                  w12,w22,w32]
    可以看出列数代表后一层神经元数目，行数代表前一层神经元数目'''
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def done_network_one():
    '''神经网络在mnist中的实现'''
    _, _, x, t = get_data()
    network = init_mnist()
    accuracy_cnt = 0
    for i in range(len(x)):
        y = predict(network, x[i])
        p = np.argmax(y)  # 返回最大值的下标
        if p == t[i]:
            accuracy_cnt += 1
    print("Accuracy:" + str(float(accuracy_cnt) / len(x)))


def done_network_pack():
    '''神经网络批处理图像mnist
    x是100*784矩阵，相当于同时对100个图像进行处理，导出的结果是100*10的矩阵
    axis代表沿着第1维方向（即行方向）寻找值最大的下标,输出结果是每一行最大值的下标矩阵中第0维是列，第1维是行'''
    _, _, x, t = get_data()
    network = init_mnist()
    batch_size = 100  # 批处理的数量
    accuracy_cnt = 0
    for i in range(0, len(x), batch_size):  # 从start到end-1步长为batch_size
        x_batch = x[i:i+batch_size]  # x[0:100],x[100,200]...
        y_batch = predict(network, x_batch)
        p = np.argmax(y_batch, axis=1)  # y中每一行的预测结果的下标，100行
        accuracy_cnt += np.sum(p == t[i:i+batch_size])
    print("Accuracy:" + str(float(accuracy_cnt) / len(x)))


def mean_squared_error(y, t):
    '''损失函数：表示神经网络的优劣程度——与真实情况之间的误差大小，比如均方差函数
    采用损失函数而非精确函数的原因：参数改变时，精确函数产生的变化不连续而损失函数产生的变化连续，这便于我们求导从而确定参数的改进方向

    如果是大量数据，加和取平均耗时过长，因此采用随机选取一小部分数据进行学习（mini-batch）
    选取实例：
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True,normalize=True,one_hot_label=True)
    train_size=x_train.shape[0]
    batch_size=3
    batch_mask=np.random.choice(train_size,batch_size)#从6000个里面选10个
    x_batch=x_train[batch_mask]
    t_batch=t_train[batch_mask]
    print(t_batch)#[[0. 0. 0. 1. 0. 0. 0. 0. 0. 0.][0. 0. 0. 0. 0. 1. 0. 0. 0. 0.][0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]]

    注意这里的t，在手写数字识别中t是one-hot形式的数组即仅有正确解为1其余皆为0
    以均方差函数为例'''
    return 0.5*np.sum((y-t)**2)


def cross_entropy_error(y, t):
    '''交叉熵误差：仅计算t中正确标签对应的测试标签概率的对数
    （由于概率小于1其对数小于0所以结果要*-1）改善一下让他可以同时处理单个数据和批数据

    如果t是标签形式，仅在每个对应图片序号下标填入对应数字则利用np.arrange(size)：生成一个0-size-1的数组
    即：return -np.sum(t*np.log(y[np.arrange(batch_size),t]+delta))/batch_size
    会生成[y[0,3],y[1,7],y[2,6]...]，即神经网络输出的n维计算结果中，第0张的正确答案是3，第1张答案是7...'''
    if y.ndim == 1:  # y是神经网络的输出
        y = y.reshape(1, y.size)
        t = t.reshape(1, y.size)
    batch_size = y.shape[0]
    delta = 1e-7  # 如果概率为0则计算log0会是负无穷，加个微小值避免这种情况
    return -np.sum(t*np.log(y+delta))/batch_size

# 开始看怎么选择参数以减少损失函数的值
# 预备知识：导数和梯度的实现


def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h)-f(x-h))/(2*h)


def tangent_line_diff(f, x):
    d = numerical_diff(f, x)
    print(d)
    y = f(x) - d*x
    return lambda t: d*t + y


def numerical_gradient_no_batch(f, x):
    '''梯度指向的方向是各点函数值减少最多的方向'''
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)
    for i in range(x.size):
        tmp_val = x[i]
        x[i] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)
        x[i] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[i] = (fxh1 - fxh2) / (2*h)
        x[i] = tmp_val  # 还原值
    return grad


def numerical_gradient(f, X):
    '''数值微分批量数据求梯度：非常耗时一般用来检查梯度的正确性'''
    if X.ndim == 1:
        return numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)  # 和X形状相同的空数组
        for i, x in enumerate(X):  # 同时列出下标与下标对象
            grad[i] = numerical_gradient_no_batch(f, x)
        return grad


def fun_test(x):
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1)


def tangent_line_gradient(f, x):  # 返回梯度函数
    d = numerical_gradient(f, x)
    print(d)
    y = f(x) - d*x
    return lambda t: d*t + y


def gradient_pic_show():
    x0 = np.arange(-2, 2.5, 0.25)
    x1 = np.arange(-2, 2.5, 0.25)
    X, Y = np.meshgrid(x0, x1)
    X = X.flatten()
    Y = Y.flatten()
    grad = numerical_gradient(fun_test, np.array([X, Y]))

    plt.figure()
    # ,headwidth=10,scale=40,color="#444444")
    plt.quiver(X, Y, -grad[0], -grad[1],  angles="xy", color="#666666")
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.grid()
    plt.legend()
    plt.draw()
    plt.show()


def gradient_descend(f, init_x, lr=0.01, step_num=100):
    '''梯度下降法调整参数—lr:更新参数的程度==学习率（这种人为调整的参数也称作超参数）,step_num：梯度下降法的重复次数
    x=x-lr*grad==一次调整
    gradient_descend(fun_test,init_x=np.array([-3.0,4.0]),lr=0.1,step_num=100)
    这里是针对fun_2函数进行梯度下降调整输入的x以达到fun_2(x)的最小值'''
    x = init_x
    for _i in range(step_num):
        grad = numerical_gradient(f, x)
        x = x-grad*lr
    return x
# 神经网络中的梯度：损失函数关于权重参数的梯度


class simpleNet:
    '''随机生成一个2*3矩阵
    predict：对其进行点乘（即神经网络中的隐藏层）再进行输出
    loss：判断输出的y矩阵中显示的各个结果概率和答案t的差别以获得损失函数'''

    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss


def done_simpleNet():
    def f(w): return net.loss(x, t)
    net = simpleNet()
    x = np.array([0.6, 0.9])
    t = np.array([1, 0, 0])
    dW = numerical_gradient(f, net.W)
    print(dW)  # W:权重参数


'''神经网络的学习
    1.mini_batch抽取数据
    2.计算损失函数对于各个权重参数的梯度大小
    3.按照梯度对权重参数进行微小改变
    4.重复1，2，3'''


class TwoLayerNet:
    '''两层神经网的实现，一层隐藏层
    W1第一层权重的形状：输入层的神经元个数*第一层神经网络的神经元个数，不包括偏置神经元b
    np.random.rand()通过本函数可以返回一个或一组服从“0~1”均匀分布的随机样本值
    np.random.randn()通过本函数可以返回一个或一组服从标准正态分布的随机样本值'''

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * \
            np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * \
            np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        return y
    # x:输入数据, t:监督数据(训练数据)

    def loss(self, x, t):  # 计算损失函数
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    # x:输入数据, t:监督数据

    def numerical_gradient(self, x, t):
        def loss_W(W): return self.loss(x, t)
        '''保存梯度的字典型变量，即numerical_gradient的返回值'''
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads


def done_mini_batch_in_TwoLayerNet():
    (x_train, t_train), (x_test, t_test) = load_mnist(
        normalize=True, one_hot_label=True)
    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    iters_num = 10000  # 适当设定循环的次数
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    iter_per_epoch = max(train_size / batch_size, 1)
    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # 计算梯度
        grad = network.numerical_gradient(x_batch, t_batch)
        #grad = network.gradient(x_batch, t_batch)

        # 更新参数——梯度下降法调整权重参数
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] = network.params[key] - \
                learning_rate * grad[key]
        # 记录学习过程
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)

        if i % iter_per_epoch == 0:  # epoch是指程序循环几次minibatch才能把所有的训练数据看完
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print("train acc, test acc | " +
                  str(train_acc) + ", " + str(test_acc))

    # 绘制图形
    _markers = {'train': 'o', 'test': 's'}
    x = np.arange(len(train_acc_list))
    plt.plot(x, train_acc_list, label='train acc')
    plt.plot(x, test_acc_list, label='test acc', linestyle='--')
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')  # 图例
    plt.show()
# 高效计算权重参数梯度的方法——误差反向传播法


class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x*y
        return out

    def backward(self, dout):
        dx = dout*self.y
        dy = dout*self.x
        return dx, dy


class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x+y
        return out

    def backward(self, dout):
        dx = dout*1
        dy = dout*1
        return dx, dy


class Relu:
    '''将输入中所有<=0的值置为0'''

    def __init__(self):
        self.mask = None  # 类似之前t标签数组，为0/1构成的数组

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0  # mask为1处会置为0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx


class Sigmoid:
    '''exp(x)的反向传播：exp(x);y=1/x的反向传播：-y**2'''

    def __init__(self):
        self.out = None  # 保存一下输出用于反向传播计算

    def forward(self, x):
        out = 1/(1+np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout*((1.0-self.out)*self.out)
        return dx


class Affine:
    '''神经网络的正向传播也称为‘仿射变换’=一次线性变换和一次平移——Affine层
    结论1）矩阵X*W对X进行求导，答案是W的转置
        关于矩阵的转置：可以将X中的元素视为变元，将X*W矩阵中每个元素针对X中的变元进行多元函数求导运算，并将结果行展开
    结论2）X和dL/dX的形状具有一致性，L是反向传播时候输出层的上游值
    结论3）针对批处理中偏置的计算，dL/dB=dL/dY的列方向求和，使得原本形状为(N*3)的dL/dY转为(3,)形状，即与B形状相同
        问题1.为何进行加和处理？为了统一形式吗？
        问题2.为何是按列进行加和？
        答：在计算L对B的偏导时，仅仅考虑B对Y的贡献，此时的上游L对Y偏导在列方向整合起来就是N*B'''

    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.original_x_shape = None  # 考虑四维情形
        self.dW = None
        self.db = None

    def forward(self, x):
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)  # 将x变成一个二维的形状（行，列为剩下的维数相乘）
        self.x = x
        out = np.dot(x, self.W)+self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        dx = dx.reshape(*self.original_x_shape)  # 还原输入数据的形状（对应张量）
        return dx


class SoftmaxWithLoss:
    ''''softmax的反向传播会得到y-t的结果，正是输出和答案的差分，前面的层可以根据这个差分的大小进行相应大小的学习'''

    def __init__(self):
        self.loss = None
        self.y = None  # softmax的输出
        self.t = None  # 监督数据==训练数据

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:  # 监督数据是one-hot-vector的情况
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        return dx


class TwoLayerNet_in_Layer:
    '''模块化操作，用记录顺序的字典记录操作的两层神经网络
    字典：keys返回所有关键字,values返回所有值,items返回所有键对'''

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * \
            np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * \
            np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # 生成层
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    # x:输入数据, t:监督数据
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)  # 返回行中最大值的下标
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # x:输入数据, t:监督数据
    def numerical_gradient(self, x, t):
        def loss_W(W): return self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 设定
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads


class SGD:
    '''随机梯度下降法（Stochastic Gradient Descent）'''

    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr*grads[key]


class Momentum:
    '''动量法更新权重参数
    v=αv-lr*grads;W=W+v（v是速度，W是权重参数）'''

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)  # 形状一样但是元素全为0

        for key in params.keys():
            self.v[key] = self.momentum*self.v[key] - self.lr*grads[key]
            params[key] += self.v[key]


class AdaGrad:
    '''根据权重参数更新幅度改变学习率，使变动大的参数的学习率逐渐减小
    学习率衰减：前期大幅度提升学快点，后面学慢点以接近最优值【这书的翻译好差劲啊】'''

    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


'''权重初始值的选择——保持输出数据的广度，避免梯度消失
1）init_weight=1则输出值的分布仅依靠其激活函数，以sigmoid函数为例，在靠近0、1时其斜率为0，会导致输出函数在0，1附近“梯度消失”问题
2）init_weight=0.01大家长得太像了导致输出函数的“表现力受限”问题（全都集中在0.5了没有意义）
3）Relu函数——He初始值（根号下2/n）S型曲线函数——Xavier初始值（根号下1/n），n是前一层神经元的数目
为调整输出函数的分布，插入对其进行正规化的层——Batch Normalization层（可以加快学习速度并使学习对权重初始值健壮）
'''

'''过拟合问题——数据多但是训练部分少，两个解决的正则化方法如下
解决办法1）权值衰减，即通过对过大的权重对应的损失函数增加与权重有关的一个量，使得越大的权重其损失函数也越大，达到快速降低大权重参数的目的
解决办法2）Dropout 随即删除隐藏层神经元抑制过拟合的出现'''

'''超参数的验证，通过在限定范围内随机取样计算其在训练数据集内部和验证数据集内部的识别精度
随机取样的代码：lr=10**np.random.uniform(low,high,size)#从一个均匀分布[low,high)中随机采样,size是输出数据的数目'''

'''卷积运算：输入数据对滤波器（一个矩阵）进行分步的矩阵运算
输入数据（H,W）滤波器（FH，FW）步幅S，填充P，输出（OH，OW）
OH=(H+2P-FH)/S+1;OW=(W+2P-FW)/S+1
如果是多维数据那么其输入数据的维数（通道数）和滤波器的维数要保持一致，运算结果是每个通道进行滤波器处理后叠加的结果
多维输入对应多维滤波器对应一维输出，如果想要多维输出，增加滤波器的“个数”!=通道数!=维数
(N,C,H,W)-(FN,C,FH,FW)->(N,FN,OH,OW)-(+FN,1,1)->(N,FN,OH,OW)
池化层：按照步幅S取其对应矩阵区域内的最大值（Max池化）一般来说输出层形状==S*S
'''
class Convolution:
    def __init__(self,W,b,stride=1,pad=0):
        self.W=W
        self.b=b
        self.stride=stride
        self.pad=pad
    def forward(self,x):
        FN,C,FH,FW=self.W.shape
        N,C,H,W=x.shape
        out_h=int(1+(H+2*self.pad-FH)/self.stride)
        out_w=int(1+(W+2*self.pad-FW)/self.stride)
        col=im2col(x,FH,FW,self.stride,self.pad)
        col_W=self.W.reshape(FN,-1).T
        out=np.dot(col,col_W)+self.b
        out=out.reshape(N,out_h,out_w,-1).transpose(0,3,1,2)
        return out