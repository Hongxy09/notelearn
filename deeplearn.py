import numpy as np
import matplotlib.pyplot as plt
'''感知机
        x1w1+x2w2+(-theta)<=0输出0
        x1w1+x2w2+(-theta)> 0输出1
        b==theta
'''
def AND(x1,x2):
    w1,w2,theta=0.5,0.5,0.7
    x=np.array([x1,x2])
    w=np.array([w1,w2])
    b=-theta
    temp=np.sum(x*w)+b
    if temp<=0:
        return 0
    return 1
def OR(x1,x2):
    w1,w2,theta=0.5,0.5,0.2
    x=np.array([x1,x2])
    w=np.array([w1,w2])
    b=-theta
    temp=np.sum(x*w)+b
    if temp<=0:
        return 0
    return 1
def NAND(x1,x2):#与非门
    w1,w2,theta=-0.5,-0.5,-0.7
    x=np.array([x1,x2])
    w=np.array([w1,w2])
    b=-theta
    temp=np.sum(x*w)+b
    if temp<=0:
        return 0
    return 1
def XOR(x1,x2):#异或门
    s1=NAND(x1,x2)
    s2=OR(x1,x2)
    return AND(s1,s2)
#简化加入激活函数
def step_fun(x):#阶跃函数(支持数组)
    return np.array(x>0,dtype=np.int)
    #判断数组中的数是否大于0并将结果用astype转化为int型
def identifify_fun(x):
    return x#恒等函数
def sigmoid(x):#s形函数
    return 1/(1+np.exp(-x))
def relu(x):
    return np.maximum(0,x)
def softmax(a):
    '''概率函数，y[0]的概率...y[1]的概率...
    yk=exp(xk)/sum-exp(xi)[1<=i<=n]
    防止指数函数运算溢出，将分子分母分别在指数内加一个系统上限数或者输入函数中的最大数'''
    c=np.max(a)
    exp_a=np.exp(a-c)
    sum_exp_a=np.sum(exp_a)
    y=exp_a/sum_exp_a
    return y
'''np.dot(矩阵点乘);np.shape(多维数组第0维元素个数，第1维元素个数...)=(3,2)三行两列'''
#矩阵实现三层神经网络
def init_network_ex():
    '''第0层两个神经元，第1层三个神经元则有3个偏置权重b（分别代表第1层三个神经元的激活阈值）
    将神经网络第0层x数值用1*2矩阵X代替，第0层神经元流向第1层神经元——第1层权重用2*3矩阵W1代替
    偏置权重用1*3矩阵B1代替，第1层神经元的输入（即隐藏层（第1，2..层）的加权和）可以用A1代替
    隐藏层的加权和经过激活函数后的数值用Z1代替
    第2层第3层类推...
    注：举例为第0层（输入层）2个神经元，第1层3个，第2层2个，第3层（输出层）2个
    '''
    network={}#{}表示字典数据类型
    #第0层到第1层的权重和偏置权重
    network['W1']=np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network['b1']=np.array([0.1,0.2,0.3])
    #第1层到第2层
    network['W2']=np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    network['b2']=np.array([0.1,0.2])
    #第2层到输出层
    network['W3']=np.array([[0.1,0.3],[0.2,0.4]])
    network['b3']=np.array([0.1,0.2])
    return network
def predict(network,x):
    W1,W2,W3=network['W1'],network['W2'],network['W3']
    b1,b2,b3=network['b1'],network['b2'],network['b3']
    a1=np.dot(x,W1)+b1#a1,a2,a3=第1层的加权信号和+偏置权重
    z1=sigmoid(a1)#加权和A经过激活函数后的信号记作Z
    a2=np.dot(z1,W2)+b2
    z2=sigmoid(a2)
    a3=np.dot(z2,W3)+b3
    #y=identifify_fun(a3)#y=a3
    y=softmax(a3)
    #输出层函数：回归问题=恒等函数，分类问题=sofmax函数
    return y
def done_network_ex():#输出神经网络输出层结果
    network=init_network_ex()
    x=np.array([1,0.5])
    y=predict(network,x)
    print(y)
    print(np.sum(y))
#mnist的导入与处理
import sys,os
import pickle
sys.path.append(os.pardir)#将父目录加入搜索路径
from dataset.mnist import load_mnist
from PIL import Image#导入图像显示模块
def img_show(img):
    '''可以打印一维数组存储的非标准化图像，若标准化则显示出来的图像是黑色
    img = np.asarray(image)图像转化为数组
    reimage=Image.fromarray(np.uint8(img))数组转化为图像
    img=img.reshape(28,28)将图片展开为原本的尺寸
    uint8：八位无符号数'''
    img = img.reshape(28, 28) # 把图像的形状变成原来的尺寸
    pil_img=Image.fromarray(np.uint8(img))#实现从array到image的转换
    pil_img.show()
def get_data():
    '''(训练图像，训练标签)(测试图像，测试标签)
    flatten：true则将图像压缩为784大小一维数组，false则展开为1*28*28三维数组
    注意此处是压缩并标准化，正常显示的图像'''
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True,normalize=True)
    return x_test, t_test,x_train,t_train
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
    x,t,_,_= get_data()
    network = init_mnist()
    accuracy_cnt = 0
    for i in range(len(x)):
        y = predict(network, x[i])
        p= np.argmax(y) #返回最大值的下标
        if p == t[i]:
            accuracy_cnt += 1
    print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
def done_network_pack():
    '''神经网络批处理图像mnist
    x是100*784矩阵，相当于同时对100个图像进行处理，导出的结果是100*10的矩阵
    axis代表沿着第1维方向（即行方向）寻找值最大的下标,输出结果是每一行最大值的下标矩阵中第0维是列，第1维是行'''
    x,t,_,_=get_data()
    network = init_mnist()
    batch_size=100#批处理的数量
    accuracy_cnt = 0
    for i in range(0,len(x),batch_size):#从start到end-1步长为batch_size
        x_batch=x[i:i+batch_size]#x[0:100],x[100,200]...
        y_batch=predict(network,x_batch)
        p=np.argmax(y_batch,axis=1)#y中每一行的预测结果的下标，100行
        accuracy_cnt+=np.sum(p==t[i:i+batch_size])
    print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
done_network_pack()