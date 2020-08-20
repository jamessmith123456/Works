# -*- coding=utf-8 -*-
# 卡尔曼滤波器
# http://www.cs.unc.edu/~welch/kalman/kalmanIntro.html

import numpy
import pylab
import matplotlib.pyplot as plt
# 这里是假设A=1，H=1的情况

# 参数初始化
n_iter = 50 #迭代次数
sz = (n_iter,)  # size of array
x = -0.37727  # truth value (typo in example at top of p. 13 calls this z)真实值
z = numpy.random.normal(x, 0.1, size=sz)  #获得一组满足正态分布的观测值 均值为x 标准差为0.1  (50,)

Q = 1e-5  # 处理方差
xhat = numpy.zeros(sz)  #滤波估计值
P = numpy.zeros(sz)  #滤波估计协方差矩阵
xhatminus = numpy.zeros(sz)  #估计值
Pminus = numpy.zeros(sz)  # a估计协方差矩阵
K = numpy.zeros(sz)  #卡尔曼增益

R = 0.1 ** 2  # 估计时的测量方差 影响滤波效果

#初始化
xhat[0] = 0.0
P[0] = 1.0

for k in range(1, n_iter):
    # 预测
    xhatminus[k] = xhat[k - 1]  # X(k|k-1) = AX(k-1|k-1) + BU(k) + W(k),A=1,BU(k) = 0
    Pminus[k] = P[k - 1] + Q  # P(k|k-1) = AP(k-1|k-1)A' + Q(k) ,A=1

    # 更新
    K[k] = Pminus[k] / (Pminus[k] + R)  # Kg(k)=P(k|k-1)H'/[HP(k|k-1)H' + R],H=1
    xhat[k] = xhatminus[k] + K[k] * (z[k] - xhatminus[k])  # X(k|k) = X(k|k-1) + Kg(k)[Z(k) - HX(k|k-1)], H=1
    P[k] = (1 - K[k]) * Pminus[k]  # P(k|k) = (1 - Kg(k)H)P(k|k-1), H=1

plt.figure()
plt.plot(z, 'k+', label='noisy measurements')  # 观测值
plt.plot(xhat, 'b-', label='a posteri estimate')  # 滤波估计值
plt.axhline(x, color='g', label='truth value')  # 真实值
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Voltage')

plt.figure()
valid_iter = range(1, n_iter)  # Pminus not valid at step 0
plt.plot(valid_iter, Pminus[valid_iter], label='a priori error estimate')
plt.xlabel('Iteration')
plt.ylabel('$(Voltage)^2$')
plt.setp(pylab.gca(), 'ylim', [0, .01])
plt.show()