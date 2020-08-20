# 利用随机森林对空气质量进行预测，通过对比分析随机森林在该领域的适用性；
# 通过完成本课题，学生将熟悉随机森林模型的基本结构、空气质量预测模型的建模流程以及随机森林在空气质量预测中的适用性
#
# 具体要求包括以下几个方面:(1)空气质量数据搜集 (2)利用随机森林对空气质量进行预测 (3)通过对比分析对模型效果进行检验

import csv
import datetime
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']

def fun(x):
    return datetime.date(*([int(i) for i in x.split('-')]))
#
# x1 = '2014-02-02'
# x2 = '2014-02-03'
# x11 = fun(x1)
# x22 = fun(x2)
# x22>x11 True

def readData(path):
    path = "F:/f0407/L1388/data/air.csv"
    with open(path,"r", encoding='utf-8') as f:
        reader = csv.reader(f)
        content = []
        for row in reader:
            content.append(list(row))
    title = content[0]
    data = content[1:]
    return title, data

def analysis(title, data):
    data_new = []
    for item in data:
        item[0] = fun(item[0]) #将时间转换为datatime格式 方便比较大小
        data_new.append(item)
    all_cities = {}
    for i in range(len(data_new)):
        if data_new[i][1] not in all_cities.keys():
            all_cities[data_new[i][1]] = []
    for i in range(len(data_new)):
        all_cities[data[i][1]].append(data_new[i])

    print("---数据集已读取完毕---")
    print("数据集各列指标含义为：",title)
    print("数据集总记录数目：",len(data))
    print("数据集总城市数目：",len(all_cities.keys()))
    return all_cities

def takeOne(elem):
    return elem[0]

def get_train_test(all_cities, prop):
    all_cities_train_test_data = {}
    for item in all_cities.keys():
        this_city = all_cities[item]
        this_city.sort(key=takeOne)
        this_city_data = []
        this_city_label = []
        for i in range(4,len(this_city)):  # 2345678
            temp_this_train = this_city[i-1][2:9]+this_city[i-2][2:9]+this_city[i-3][2:9]
            if '' not in temp_this_train:
                this_train = [float(item) for item in temp_this_train]
                this_label = float(this_city[i][2])
                this_city_data.append(this_train)
                this_city_label.append(this_label)
        this_city_data = np.array(this_city_data)
        this_city_label = np.array(this_city_label)
        this_city_train_data = this_city_data[0:int(prop*this_city_data.shape[0]),:]
        this_city_train_label = this_city_label[0:int(prop * this_city_data.shape[0])]
        this_city_test_data = this_city_data[int(prop*this_city_data.shape[0])+1:,:]
        this_city_test_label = this_city_label[int(prop * this_city_data.shape[0])+1:]
        all_cities_train_test_data[item] = [this_city_train_data, this_city_train_label, this_city_test_data, this_city_test_label]

    return all_cities_train_test_data






if __name__ == '__main__':
    path = "F:/f0407/L1388/data/air.csv"
    title, data = readData(path)
    all_cities = analysis(title, data)
    all_cities_train_test_data = get_train_test(all_cities, 0.8)

    #下面就某个具体的城市来进行随机森林预测 阿坝州 常熟 鄂尔多斯 济南  吉林  广州 盐城 上海 苏州
    X_train, y_train, X_test, y_test = all_cities_train_test_data['苏州']
    rf = RandomForestRegressor(n_estimators=500, random_state=0)
    rf.fit(X_train, y_train)
    y_hat = rf.predict(X_test)
    print(rf.score(X_train, y_train))
    print(rf.score(X_test, y_test))
    plt.figure(figsize=(30, 12))
    plt.plot(y_test, '-r', label='真实值', marker='o')
    plt.plot(y_hat, '-g', label='预测值', marker='D')
    plt.legend()
    plt.title("苏州市随机森林预测结果", fontsize=20)
    plt.savefig("第一个数据集苏州随机森林图.png")
    plt.show()

    X_train, y_train, X_test, y_test = all_cities_train_test_data['上海']
    rf = RandomForestRegressor(n_estimators=500, random_state=0)
    rf.fit(X_train, y_train)
    y_hat = rf.predict(X_test)
    print(rf.score(X_train, y_train))
    print(rf.score(X_test, y_test))
    plt.figure(figsize=(30, 12))
    plt.plot(y_test, '-r', label='真实值', marker='o')
    plt.plot(y_hat, '-g', label='预测值', marker='D')
    plt.legend()
    plt.title("上海市随机森林预测结果", fontsize=20)
    plt.savefig("第一个数据集上海随机森林图.png")
    plt.show()

    X_train, y_train, X_test, y_test = all_cities_train_test_data['北京']
    rf = RandomForestRegressor(n_estimators=500, random_state=0)
    rf.fit(X_train, y_train)
    y_hat = rf.predict(X_test)
    #rf.score(X_test)
    print(rf.score(X_train, y_train))
    print(rf.score(X_test, y_test))
    plt.figure(figsize=(30, 12))
    plt.plot(y_test, '-r', label='真实值', marker='o')
    plt.plot(y_hat, '-g', label='预测值', marker='D')
    plt.legend()
    plt.title("北京市随机森林预测结果", fontsize=20)
    plt.savefig("第一个数据集北京随机森林图.png")
    plt.show()


