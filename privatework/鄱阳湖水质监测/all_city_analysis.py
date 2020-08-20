import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from pylab import *

mpl.rcParams['font.sans-serif'] = ['SimHei']

def value_to_level(AQI):
    if AQI>=0 and AQI<=50:
        return '一级'
    elif AQI>=51 and AQI<=100:
        return '二级'
    elif AQI>=101 and AQI<=150:
        return '三级'
    elif AQI>=151 and AQI<=200:
        return '四级'
    elif AQI>=201 and AQI<=300:
        return '五级'
    else:
        return '六级'


if __name__ == '__main__':
    data = pd.read_csv("./data/data.csv")
    data.info()

    #缺失值处理
    print(data.Precipitation.skew())  # skew求偏态系数
    sns.distplot(data.Precipitation.dropna())  # dropna()删掉缺失的数据，displot不能处理缺失的数据
    plt.show()
    #均值/中值填充
    data.fillna({'Precipitation': data['Precipitation'].median()}, inplace=True)
    #异常值处理
    plt.figure(figsize=(15, 4))
    plt.xticks(rotation=45, fontsize=15)
    sns.boxplot(data=data)
    #plt.show()

    #三个特征值存在明显的异常值，利用箱体图进行处理
    t = data.copy()
    for k in t:
        if pd.api.types.is_numeric_dtype(t[k]):
            o = t[k].describe()
            IQR = o["75%"] - o["25%"]
            lower = o["25%"] - 1.5 * IQR
            upper = o["75%"] + 1.5 * IQR
            t[k][t[k] < lower] = lower
            t[k][t[k] > lower] = upper
    plt.figure(figsize=(15, 4))
    plt.xticks(rotation=45, fontsize=15)
    sns.boxplot(data=t)
    #plt.show()

    #重复值处理
    # 发现重复值
    print(data.duplicated().sum())
    # 查看哪些记录出现了重复值
    data[data.duplicated()]

    #空气质量排名
    plt.subplot(231)
    # 空气最好前五/最差五个的城市
    best = data[['City', 'AQI']].sort_values(by=['AQI'])
    print(best.head(5))
    plt.xticks(rotation=45)
    plt.title('空气质量最好排名')
    sns.barplot(x='City', y='AQI', data=best.head(5))
    plt.subplot(233)
    low = data[['City', 'AQI']].sort_values(by=['AQI'], ascending=False)
    print(best.head(5))
    plt.xticks(rotation=45)
    plt.title('空气质量最差排名')
    sns.barplot(x='City', y='AQI', data=low.head(5))
    plt.savefig("第二个数据集城市空气质量排名图.png")
    #plt.show()
    #最好的为：韶关，南平，梅州，基隆，三明
    #最差的为：北京，朝阳，保定，锦州，焦作

    #全国空气质量等级统计
    level=data['AQI'].apply(value_to_level)
    print(level.value_counts())
    sns.countplot(x=level,order=['一级','二级','三级','四级','五级','六级'])

    #全国空气指数分布
    sns.scatterplot(x='Longitude', y='Latitude', hue='AQI', palette=plt.cm.RdYlGn_r, data=data)

    # 定义总体数据
    total = np.random.normal(loc=30, scale=80, size=10000)
    # 创建均值数据
    mean = np.zeros(1000)
    for i in range(len(mean)):
        mean[i] = np.random.choice(total, size=64, replace=False).mean()
    print('样本均值：', mean.mean())
    print('样本标准差：', mean.std())
    print('偏度：', pd.Series(mean).skew())
    sns.distplot(mean)

    # 定义标准差
    scale = 50
    # 定义数据
    x = np.random.normal(0, scale, size=100000)
    # 定义标准差的倍数，倍数从1到3
    for times in range(1, 4):
        y = x[(x >= -times * scale) & (x <= times * scale)]
        print(f'{times}倍标准差：')
        print(f'{len(y) * 100 / len(x)}%')


    r=stats.ttest_1samp(data['AQI'],72)
    print('t值',r.statistic)
    print('p值',r.pvalue)
    #计算结果：t值1.393763441074581 p值0.16435019471704654。P值大于0.05，故在显著度为0.05 检验下无法拒绝原假设。

    # 计算全国平均空气质量指数均值
    n = len(data)
    df = n - 1
    left = stats.t.ppf(0.025, df=df)
    right = stats.t.ppf(0.975, df=df)
    print(left, right)
    mean = data['AQI'].mean()
    std = data['AQI'].std()
    mean + left * (std / np.sqrt(n)), mean + right * (std / np.sqrt(n))
    # 计算结果：-1.9673585853224684 1.967358585322468
    # (70.6277615675309, 80.0409690826239)
    # 结论：全国空气质量指数所在区间大概在70.63-80.04之间，置信度为95%

    # 对空气质量进行预测
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split

    X = data.drop(['City', 'AQI'], axis=1)
    y = data['AQI']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_hat = lr.predict(X_test)
    print(lr.score(X_train, y_train))
    print(lr.score(X_test, y_test))
    plt.figure(figsize=(15, 6))
    plt.plot(y_test.values, '-r', label='真实值', marker='o')
    plt.plot(y_hat, '-g', label='预测值', marker='D')
    plt.legend()
    plt.title("线性回归预测结果", fontsize=20)
    from sklearn.metrics import mean_absolute_error
    print(mean_absolute_error(y_hat, y_test.values))
    plt.savefig("第二个数据集线性回归预测图.png")
    plt.show()

    from sklearn.ensemble import RandomForestRegressor

    rf = RandomForestRegressor(n_estimators=500, random_state=0)
    rf.fit(X_train, y_train)
    y_hat = rf.predict(X_test)
    print(rf.score(X_train, y_train))
    print(rf.score(X_test, y_test))
    plt.figure(figsize=(15, 6))
    plt.plot(y_test.values, '-r', label='真实值', marker='o')
    plt.plot(y_hat, '-g', label='预测值', marker='D')
    plt.legend()
    plt.title("随机森林预测结果", fontsize=20)
    from sklearn.metrics import mean_absolute_error
    print(mean_absolute_error(y_hat, y_test.values))
    plt.savefig("第二个数据集随机森林预测图.png")
    plt.show()
    #总结
    # 空气总体质量南部好于北部，西部好于东部；沿海好于内地。
    # 降雨量与纬度对空气质量影响较大，存在一个可以相关系数，即沿海。
    # 我国城市平均质量指数在70.63到80.04之间，概率高达95 %。
    # 通过历史数据，可以进行预测。
    # 虽然我国整体空气质量较好，但是一级的也不是最多的，且四五级占的比例也不低，因此需要加快整治步伐，同时大力宣传，提升国民的环保意识。
