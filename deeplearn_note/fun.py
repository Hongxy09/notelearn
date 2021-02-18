import matplotlib.pyplot as plt
import numpy as np
import pickle
from collections import OrderedDict
from dataset.mnist import load_mnist
import os
import sys
sys.path.append(os.pardir)


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


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    input_data : 由(数据量, 通道, 高, 长)的4维数组构成的输入数据
    filter_h : 滤波器的高
    filter_w : 滤波器的长
    stride : 步幅
    pad : 填充

    Returns
    -------
    col : 2维数组
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    print("out_h:", out_h, " out_w:", out_w)
    img = np.pad(input_data, [(0, 0), (0, 0),
                              (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):  # 这一层for循环是扫描输入数据中当前行感知野包含的数据并放入col中
            x_max = x + stride*out_w  # 滤波器开始扫描第一层的第一个感知野并放入col中
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
        print("col:", col)
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    # 将每个输出的小矩阵展开为一列并拼在一起后输出
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    col :
    input_shape : 输入数据的形状（例：(10, 1, 28, 28)）
    filter_h :
    filter_w
    stride
    pad

    Returns
    -------

    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h,
                      filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]


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


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        self.C = None

        # 权重和偏置参数的梯度
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW, = self.W.shape
        N, C, H, W = x.shape
        out_h = (H+2*self.pad-FH)/self.stride+1
        out_w = (W+2*self.pad-FW)/self.stride+1

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1)  # 滤波器展开
        out = np.dot(col, col_W)+self.b

        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        self.C = C
        self.x = x
        self.col = col
        self.col_W = col_W
        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(H+2*self.pad-self.pool_h)/self.stride+1
        out_w = int(W+2*self.pad-self.pool_w)/self.stride+1
        # 展开
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)
        # max
        out = np.max(col, axis=1)  # 取每一行的最大值
        # 转回为N个三维数组
        out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        arg_max = np.argmax(col, axis=1)
        self.x = x
        self.arg_max = arg_max
        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size),
             self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h,
                    self.pool_w, self.stride, self.pad)

        return dx
