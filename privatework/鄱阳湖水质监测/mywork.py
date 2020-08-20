# -*- coding: utf-8 -*-

import SCN as scn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
from sklearn.metrics import confusion_matrix
import scipy.signal as ss
import os
import xlrd
import copy
plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）

def transfer(X):
    sh = np.shape(X)
    m = sh[0]
    A = np.zeros((m))
    for i in range(m):
        A[i] = np.argmax(X[i, :])
    return A

def get_min_max(X):
    new_x = copy.deepcopy(X)
    Xmax = np.max(X,1)
    Xmin = np.min(X,1)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            new_x[i,j] = (X[i,j]-Xmin[j])/(Xmax[j]-Xmin[j])
    return new_x

def get_data(path):
    all_file = os.listdir(path)
    all_x = []
    all_y = []
    for item in all_file:
        temp_file = os.path.join(path,item)
        workbook = xlrd.open_workbook(temp_file)
        sheet_num = workbook.nsheets
        names = workbook.sheet_names()
        print("共有{}个工作表，表名分别是：{}".format(sheet_num, names))
        sales = workbook.sheet_by_name("LC02")
        sales_rows = sales.nrows  # 获取"sales"工作表中的总行数
        sales_cols = sales.ncols  # 获取"sales"工作表中的总列数
        print("sales：{}行×{}列".format(sales_rows, sales_cols))

        all_col = []
        for j in range(sales_cols):
            all_col.append(str(sales.cell(0,j).value))

        my_feature_name = ['溶解氧','pH','磷酸盐','亚硝酸盐氮','硝酸盐氮','氨态氮']
        my_target_name = ['总磷']
        my_feature_name_index = [all_col.index(item) for item in my_feature_name]
        my_target_name_index = [all_col.index(item) for item in my_target_name]

        temp_x = []
        temp_y = []
        for i in range(3, sales_rows):
            try:
                temp_temp = []
                temp_temp.append(float(sales.cell(i, my_feature_name_index[0]).value))
                temp_temp.append(float(sales.cell(i, my_feature_name_index[1]).value))
                temp_temp.append(float(sales.cell(i, my_feature_name_index[2]).value))
                temp_temp.append(float(sales.cell(i, my_feature_name_index[3]).value))
                temp_temp.append(float(sales.cell(i, my_feature_name_index[4]).value))
                temp_temp.append(float(sales.cell(i, my_feature_name_index[5]).value))
                temp_y.append(float(sales.cell(i, my_target_name_index[0]).value))
                temp_x.append(temp_temp)
            except:
                pass

        all_x.extend(temp_x)
        all_y.extend(temp_y)

    all_x = np.array(all_x)
    all_y = np.array(all_y)
    all_y = all_y[:,np.newaxis]
    return all_x, all_y

def get_mse(Y,Y2):
    result = 0
    for i in range(Y.shape[0]):
        result += (Y[i,0]-Y2[i,0])**2
    result = result/Y.shape[0]
    return result

if __name__ == '__main__':
    path = './data/'
    all_x, all_y = get_data(path)
    all_x2 = get_min_max(all_x)
    X = all_x2[0:200,:]
    T = all_y[0:200,:]
    X2 = all_x2[200:259,:]
    T2 = all_y[200:259,:]

    L_max = 250
    tol = 0.001
    T_max = 350
    Lambdas = [0.5, 1, 5, 10, 30, 50, 200, 150, 200, 250]
    nB = 1
    r = [0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999]

    M = scn.SCN(L_max, tol, T_max, Lambdas, nB, r)

    per_Error = M.Regression(X, T)
    print(M)


    O2 = scn.SCN.GetOutput(X2, T2, 1)

    fig4 = plt.figure()
    ax4 = fig4.add_subplot(1, 1, 1)
    ax4.plot(T2, color='b', linestyle='--', label='Test Target')
    ax4.plot(O2, color='r', linestyle='-.', label='Model Output')
    ax4.set_xlim()
    ax4.set_ylim()
    plt.title("随机向量函数连接神经网络总磷浓度预测-100节点")
    plt.grid(True)
    plt.legend()
    plt.show()

    mse = get_mse(T2, O2)
    print("MSE误差为：",mse)