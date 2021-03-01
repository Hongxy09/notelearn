from dataset.mnist import load_mnist
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pickle
sys.path.append(os.pardir)
'''
1.输入数据->权重加偏置进行计算->激活函数进行整理->输出下一层的值
2.输出层结果与标签计算损失函数->根据损失函数计算其相对权重的梯度->对权重进行对应的变大/变小操作，变化幅度对应梯度大小
3.用得到的新权重重复1，2操作
'''


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W)+self.b

        return out

    def backward(self, dout):
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        dx = np.dot(dout, self.W.T)

        return dx


def softmax(x):
    if x.ndim == 2:
        x = x.T  # x的转置
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    x = x - np.max(x)  # 溢出对策

    return np.exp(x) / np.sum(np.exp(x))


def cross_entropy_error(y, t):
    if y.ndim == 1:  # y是神经网络的输出
        y = y.reshape(1, y.size)
        t = t.reshape(1, y.size)
    batch_size = y.shape[0]
    delta = 1e-7  # 微小值避免负无穷

    return -np.sum(t*np.log(y+delta))/batch_size


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None  # 训练数据

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout):
        batch_size = self.t.shape[0]
        dx = (self.y-self.t)/batch_size

        return dx


class BatchNormalization:
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None  # Conv层的情况下为4维，全连接层的情况下为2维

        # 测试时使用的平均值和方差
        self.running_mean = running_mean
        self.running_var = running_var

        # backward时使用的中间数据
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)
        out = self.__forward(x, train_flg)

        return out.reshape(*self.input_shape)

    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std

            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * \
                self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * \
                self.running_var + (1-self.momentum) * var
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))

        out = self.gamma * xn + self.beta
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size

        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 **
                                 self.iter) / (1.0 - self.beta1**self.iter)

        for key in params.keys():
            #self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
            #self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])

            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)

            # unbias_m += (1 - self.beta1) * (grads[key] - self.m[key]) # correct bias
            # unbisa_b += (1 - self.beta2) * (grads[key]*grads[key] - self.v[key]) # correct bias
            #params[key] += self.lr * unbias_m / (np.sqrt(unbisa_b) + 1e-7)


class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


class Network_simple:
    def __init__(self, input_size, hidden_size_list, output_size,
                 activation='relu', weight_init_std='relu', weight_decay_lambda=0,
                 use_dropout=False, dropout_ration=0.5, use_batchnorm=False):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.use_dropout = use_dropout
        self.weight_decay_lambda = weight_decay_lambda
        self.use_batchnorm = use_batchnorm
        self.params = {}

        # 初始化权重
        self.__init_weight(weight_init_std)

        # 生成层
        activation_layer = {'relu': Relu}
        self.layers = {}
        for idx in range(1, self.hidden_layer_num+1):
            self.layers['Affine' + str(idx)] = Affine(
                self.params['W' + str(idx)], self.params['b' + str(idx)])
            if self.use_batchnorm:
                self.params['gamma' +
                            str(idx)] = np.ones(hidden_size_list[idx-1])
                self.params['beta' +
                            str(idx)] = np.zeros(hidden_size_list[idx-1])
                self.layers['BatchNorm' + str(idx)] = BatchNormalization(
                    self.params['gamma' + str(idx)], self.params['beta' + str(idx)])

            self.layers['Activation_function' +
                        str(idx)] = activation_layer[activation]()

            # if self.use_dropout:
            #     self.layers['Dropout' + str(idx)] = Dropout(dropout_ration)

        idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = Affine(self.params['W' +
                                                              str(idx)], self.params['b' + str(idx)])

        self.lastLayer = SoftmaxWithLoss()

    def __init_weight(self, weight_init_std):
        """设定权重的初始值
        Parameters
        ----------
        weight_init_std : 指定权重的标准差（e.g. 0.01）
            指定'relu'或'he'的情况下设定“He的初始值”
            指定'sigmoid'或'xavier'的情况下设定“Xavier的初始值”
        """
        all_size_list = [self.input_size] + \
            self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            str(weight_init_std).lower() in ('relu', 'he')
            scale = np.sqrt(2.0 / all_size_list[idx - 1])
            self.params['W' + str(idx)] = scale * \
                np.random.randn(all_size_list[idx-1], all_size_list[idx])
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])

    def predict(self, x, train_flg=False):
        for key, layer in self.layers.items():
            if "BatchNorm" in key:
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)

        return x

    def loss(self, x, t, train_flg=False):
        y = self.predict(x, train_flg)
        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params['W' + str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W**2)
        # 增加了L2范数
        return self.lastLayer.forward(y, t) + weight_decay

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t)/float(x.shape[0])
        return accuracy

    def gradient(self, x, t):
        # forward
        self.loss(x, t, train_flg=True)
        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        return grads

    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, key in enumerate(['Affine1', 'Affine2']):
            self.layers[key].W = self.params['W' + str(i+1)]
            self.layers[key].b = self.params['b' + str(i+1)]


class Trainer:
    def __init__(self, network, x_train, t_train, x_test, t_test, epochs=20,
                 mini_batch_size=100, optimizer='Adam', optimizer_param={'lr': 0.01},
                 evaluate_sample_num_per_epoch=None, verbose=True):
        self.network = network
        self.verbose = verbose
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs
        self.batch_size = mini_batch_size
        # 评估每个周期的样本数
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch
        # optimzer
        optimizer_class_dict = {'sgd': SGD, 'adam': Adam}
        self.optimizer = optimizer_class_dict[optimizer.lower()](
            **optimizer_param)

        self.train_size = x_train.shape[0]
        self.iter_per_epoch = max(self.train_size / mini_batch_size, 1)
        self.max_iter = int(epochs * self.iter_per_epoch)
        self.current_iter = 0
        self.current_epoch = 0

        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

    def train_step(self):
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]

        grads = self.network.gradient(x_batch, t_batch)
        self.optimizer.update(self.network.params, grads)

        loss = self.network.loss(x_batch, t_batch)
        self.train_loss_list.append(loss)
        if self.verbose:
            print("train loss:" + str(loss))

        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1

            x_train_sample, t_train_sample = self.x_train, self.t_train
            x_test_sample, t_test_sample = self.x_test, self.t_test
            if not self.evaluate_sample_num_per_epoch is None:
                t = self.evaluate_sample_num_per_epoch
                x_train_sample, t_train_sample = self.x_train[:t], self.t_train[:t]
                x_test_sample, t_test_sample = self.x_test[:t], self.t_test[:t]

            train_acc = self.network.accuracy(x_train_sample, t_train_sample)
            test_acc = self.network.accuracy(x_test_sample, t_test_sample)
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)

            if self.verbose:
                print("=== epoch:" + str(self.current_epoch) + ", train acc:" +
                      str(train_acc) + ", test acc:" + str(test_acc) + " ===")
        self.current_iter += 1

    def train(self):
        for _i in range(self.max_iter):
            self.train_step()

        test_acc = self.network.accuracy(self.x_test, self.t_test)

        if self.verbose:
            print("=============== Final Test Accuracy ===============")
            print("test acc:" + str(test_acc))


class done_network:
    def __init__(self):
        (self.x_train, self.t_train), (self.x_test,
                                       self.t_test) = load_mnist(one_hot_label=True)
        self.network = Network_simple(input_size=784, hidden_size_list=[100], output_size=10,
                                      use_batchnorm=False, activation='relu')
        self.trainer = Trainer(self.network, self.x_train, self.t_train, self.x_test, self.t_test,
                               epochs=20, mini_batch_size=100, optimizer='Adam', optimizer_param={'lr': 0.01},
                               evaluate_sample_num_per_epoch=1000)
        self.epochs = 20

    def done(self):
        self.trainer.train()
        self.network.save_params("params.pkl")
        print("Saved!")

    def draw(self):
        done_network().done()
        markers = {'train': 'o', 'test': 's'}
        x = np.arange(self.epochs)
        plt.plot(x, self.trainer.train_acc_list,
                 marker='o', label='train', markevery=2)
        plt.plot(x, self.trainer.test_acc_list,
                 marker='s', label='test', markevery=2)
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.ylim(0, 1.0)
        plt.legend(loc='lower right')
        plt.show()
done_network().draw()