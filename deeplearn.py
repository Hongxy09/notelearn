import numpy as np
import matplotlib.pyplot as plt
'''感知机x1w1+x2w2+(-theta)<=0  0
        x1w1+x2w2+(-theta)>0    1
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
#输出
test_x1=[0,1,0,1]
test_x2=[0,0,1,1]
for i in range(4):
    print(test_x1[i],test_x2[i])
    print(XOR(test_x1[i],test_x2[i]))
