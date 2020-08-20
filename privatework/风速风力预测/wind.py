# 几种风速判别：风速（启动风速 标准风速） 10m/s  风向(0-360)  最终用折线图和扇形图展示出来
# 瞬时风速/平均风速（原始数据应该表达的是瞬时风速，平均风速需要自己求）
#
# 全年360天 每天 7 14 21三个时刻  早晚风速大 中午风速小 春秋季风速大 夏季最小 冬季次之

# 现模拟数据:
#               年/风速均值           7均值/方差    14均值/方差   21均值/方差
# 1月(冬季)       11.5m/s             9.4/3.2       6.9/1.9     10.3/3.1
# 2月（冬季）      9.5m/s              7.7/2.8       6.7/1.5      7.8/2.3
# 3月（春季）      9.2m/s              7.7/1.9        6.4/1.4     7.9/1.8
# 4月（春季）      10.8m/s             10.8/2.4       9.7/1.8     11.2/2.3
# 5月（春季）      12.6m/s             12.4/2.6       10.5/1.7    12.8/3.6
# 6月（春季）      13.6m/s             13.0/3.2       10.8/1.9    14.4/4.8
# 7月（夏季）      10.8m/s             6.2/2.4       7.8/1.7     8.4/1.3
# 8月（夏季）      8.6m/s              4.6/2.5        5.2/2.1     5.5/1.1
# 9月（秋季）      10.6m/s             9.2/2.3       8.8/1.8     8.6/1.3
# 10月（秋季）     11.3m/s             10.1/2.4       8.6/1.7     11.4/2.3
# 11月（秋季）     11.9m/s             10.9/2.8       7.3/1.6     11.2/2.1
# 12月（冬季）     12.4m/s             11.6/3.4       10.8/1.9    12.3/1.9


import numpy as np
import copy
import matplotlib.pyplot as plt
import pandas as pd
import math
from matplotlib.ticker import FuncFormatter
import random
import xlwt
import xlrd

random.seed(6)
plt.rcParams['font.sans-serif'] = ['SimHei']   #设置简黑字体
plt.rcParams['axes.unicode_minus'] = False
def get_data(month=None, month_number=None):
    parameters = [[[4.4,10.2],[6.9,23.9],[10.3,26.1]],[[7.7,10.8],[6.7,24.5],[7.8,25.3]],\
                    [[3.7,10.9],[6.4,24.4],[7.9,24.8]],[[10.8,10.4],[9.7,23.8],[11.2,25.3]],\
                    [[6.4,10.6],[10.5,24.7],[12.8,24.6]],[[13.0,10.2],[10.8,23.9],[14.4,25.8]],\
                    [[3.2,10.4],[7.8,24.7],[8.4,24.1]],[[4.6,10.5],[5.2,25.1],[5.5,26.1]],\
                    [[4.2,10.3],[8.8,23.8],[8.6,24.3]],[[10.1,10.4],[8.6,25.7],[11.4,26.3]],\
                    [[5.9,10.8],[7.3,24.6],[11.2,25.1]],[[11.6,10.4],[10.8,24.9],[12.3,25.9]],\
                  ]
    all_month_7 = []
    all_month_14 = []
    all_month_21 = []
    if month!=None:
        item = parameters[month]
        record_7 = [round(item,2) for item in list(map(abs, list(np.random.normal(loc=item[0][0], scale=item[0][1], size=300))))]
        record_14 = [round(item,2) for item in list(map(abs, list(np.random.normal(loc=item[1][0], scale=item[1][1], size=300))))]
        record_21 = [round(item,2) for item in list(map(abs, list(np.random.normal(loc=item[2][0], scale=item[2][1], size=300))))]
        all_month_7.append(record_7)
        all_month_14.append(record_14)
        all_month_21.append(record_21)

    if month_number!=None:
        for i in range(month_number):
            item = month_number[i]
            record_7 = []
            record_14 = []
            record_21 = []
            record_7 = [round(item,2) for item in list(map(abs, list(np.random.normal(loc=item[0][0], scale=item[0][1], size=300))))]
            record_14 = [round(item,2) for item in list(map(abs, list(np.random.normal(loc=item[1][0], scale=item[1][1], size=300))))]
            record_21 = [round(item,2) for item in list(map(abs, list(np.random.normal(loc=item[2][0], scale=item[2][1], size=300))))]
            all_month_7.append(record_7)
            all_month_14.append(record_14)
            all_month_21.append(record_21)
    all_data = [all_month_7, all_month_14, all_month_21]
    return all_data

def analysis_1(all_data):
    mean = []
    std = []
    all_7 = []
    all_14 = []
    all_21 = []
    for j in range(len(all_data[0])):
        for k in range(len(all_data[0][j])):
            all_7.append(all_data[0][j][k])
    for j in range(len(all_data[1])):
        for k in range(len(all_data[1][j])):
            all_14.append(all_data[1][j][k])
    for j in range(len(all_data[2])):
        for k in range(len(all_data[2][j])):
            all_21.append(all_data[2][j][k])
    mean = [np.mean(all_7), np.mean(all_14), np.mean(all_21)]
    std = [np.std(all_7), np.std(all_14), np.std(all_21)]
    print("早7点风速均值：{}m/s".format(round(np.mean(all_7),2)))
    print("午14点风速均值：{}m/s".format(round(np.mean(all_14),2)))
    print("晚21点风速均值：{}m/s".format(round(np.mean(all_21),2)))

    print("早7点风速标准差：{}".format(round(np.std(all_7),2)))
    print("午14点风速标准差：{}".format(round(np.std(all_14), 2)))
    print("晚21点风速标准差：{}".format(round(np.std(all_21), 2)))
    return mean, std

def analysis_2(all_data):
    wind_dire = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    all_data_new = copy.deepcopy(all_data)
    all_data_wind = copy.deepcopy(all_data)
    all_data_wind2 = copy.deepcopy(all_data)
    for i in range(len(all_data)): #i遍历早中晚
        for j in range(len(all_data[i])): #j遍历多少个月 咱们这里就是一个月的情况
            temp = all_data[i][j] #每天10个值 30天共300个值
            temp_new = []
            temp_wind = []
            temp_wind_cor = []
            for k in range(30): #k
                x = np.arange(10)*6
                y = np.array(temp[k*10:(k+1)*10])
                #3次高阶拟合
                f1 = np.polyfit(x, y, 3)
                x_new = np.arange(60)
                y_new = np.polyval(f1, x_new)
                for l in range(y_new.shape[0]):
                    temp_new.append(abs(y_new[l]))
                    cor_num = np.random.randint(0,360)
                    if random.random()<=0.6 and cor_num>180:
                        cor_num = cor_num - 90
                    temp_wind.append(cor_num)
                    temp_wind_cor.append(wind_dire[math.floor(cor_num/22.5)])
            all_data_new[i][j] = temp_new
            all_data_wind[i][j] = temp_wind
            all_data_wind2[i][j] = temp_wind_cor

    time = [7,14,21]
    book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet = book.add_sheet('data', cell_overwrite_ok=True)
    ii = 0
    sheet.write(ii, 0, '日期')
    sheet.write(ii, 1, '风速')
    sheet.write(ii, 2, '风向')
    ii +=1
    for i in range(len(all_data_new)):
        for j in range(len(all_data_new[i])):
            temp = all_data_new[i][j]
            for k in range(30):
                for l in range(10):
                    str1 = "{:2}号 {:2}时 {:2}分".format(k, time[i], l * 6)
                    str2 = "{:4}m/s".format(round(all_data_new[i][j][k * 10 + l],2))
                    str3 = "{:4}°".format(all_data_wind[i][j][k * 10 + l])
                    sheet.write(ii, 0, str1)
                    sheet.write(ii, 1, str2)
                    sheet.write(ii, 2, str3)
                    ii+=1
    book.save('C:/Users/1299716045/Desktop/abc/test.xls')
    with open('data.txt', 'w') as file_handle:  # .txt可以不自己新建,代码会自动新建
        for i in range(len(all_data_new)):
            for j in range(len(all_data_new[i])):
                temp = all_data_new[i][j]
                for k in range(30):
                    for l in range(10):
                        data_moment = "{:2}号 {:2}时 {:2}分:  {:4}m/s  {:4}° {:4}".format(k, time[i], l*6, round(all_data_new[i][j][k*10+l],2), all_data_wind[i][j][k*10+l], all_data_wind2[i][j][k*10+l])
                        file_handle.write(data_moment)  # 写入
                        file_handle.write('\n')
    file_handle.close()
    return all_data_new, all_data_wind, all_data_wind2

def analysis_3(all_data, all_data_wind, all_data_wind2):
    min_wind = 10000
    max_wind = 0
    all_data_new2 = copy.deepcopy(all_data)
    #下面统计各个时刻的值 [2,3] [3,4] [8,13] [>25]
    qidong_min = 2 #启动风速
    qidong_max = 3
    qieru_min = 3 #切入风速
    qieru_max = 4
    eding_min = 8 #额定风速
    eding_max = 13
    jixian_min = 25 #极限风速
    jixian_max = 30
    gongzuo_min = 2 #工作风速
    gongzuo_max = 25
    time = [7,14,21]
    qidong_num_all = 0
    qieru_num_all = 0
    eding_num_all = 0
    jixian_num_all = 0
    gongzuo_all_num = 0
    wuyong_all_num = 0
    all_num = 0
    for i in range(len(all_data)):
        for j in range(len(all_data[i])):
            temp = all_data[i][j]
            for k in range(30):
                one_day = temp[k*60:(k+1)*60]
                qidong_num = 0
                qieru_num = 0
                eding_num = 0
                gongzuo_num = 0
                jixian_num = 0
                wuyong_num = 0
                for l in range(len(one_day)):
                    if one_day[l]<min_wind:
                        min_wind = one_day[l]
                    if one_day[l]>max_wind:
                        max_wind = one_day[l]
                    if qidong_min<=one_day[l]<=qidong_max:
                        qidong_num+=1
                    elif qieru_min<=one_day[l]<=qieru_max:
                        qieru_num+=1
                    elif eding_min<=one_day[l]<=eding_max:
                        eding_num+=1
                    elif gongzuo_min<=one_day[l]<=gongzuo_max:
                        gongzuo_num+=1
                    elif jixian_min<=one_day[l]<=jixian_max:
                        jixian_num+=1
                    else:
                        wuyong_num += 1
                print("{}号--{}时：启动风速{}次，切入风速{}次，额定风速{}次，工作风速{}次".format(k+1, time[i], qidong_num, qieru_num, eding_num, gongzuo_num))
                qidong_num_all += qidong_num
                qieru_num_all += qieru_num
                eding_num_all += eding_num
                jixian_num_all += jixian_num
                gongzuo_all_num += gongzuo_num
                wuyong_all_num += wuyong_num
                all_num += 1
            all_data_new2[i][j] = [qidong_num, qieru_num, eding_num, gongzuo_num]
    #绘制风速区间饼图
    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))
    recipe = ["启动风速2-3m/s:" + str(qidong_num_all),
              "切入风速3-4m/s:" + str(qieru_num_all),
              "额定风速8-13m/s:" + str(eding_num_all),
              "极限风速25-30m/s:" + str(jixian_num_all),
              "不可利用风速<3/>30m/s:" + str(wuyong_all_num),]
    data = [qidong_num_all, qieru_num_all, eding_num_all, jixian_num_all, wuyong_all_num]
    wedges, texts = ax.pie(data, wedgeprops=dict(width=0.5), startangle=-40)
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    kw = dict(xycoords='data', textcoords='data', arrowprops=dict(arrowstyle="-"),
              bbox=bbox_props, zorder=0, va="center")
    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1) / 2. + p.theta1  # 锁定扇形夹角的中间位置，对应的度数为ang
        y = np.sin(np.deg2rad(ang))  # np.sin()求正弦
        x = np.cos(np.deg2rad(ang))  # np.cos()求余弦
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)  # 参数connectionstyle用于控制箭头连接时的弯曲程度
        kw["arrowprops"].update({"connectionstyle": connectionstyle})  # 将connectionstyle更新至参数集kw的参数arrowprops中
        ax.annotate(recipe[i], size=8, xy=(x, y), xytext=(1.5 * np.sign(x), 1.5 * y),
                    horizontalalignment=horizontalalignment, **kw)
    ax.set_title("风速区间统计图")
    plt.savefig("风速区间统计图")
    plt.close()
    #绘制早中晚直方统计图
    morning = 0
    afternoon = 1
    night = 2
    all_days_morning_mean = []
    for j in range(len(all_data[morning])):
        temp = all_data[morning][j]
        for k in range(30):
            this_day = np.mean(temp[k*60:(k+1)*60])
            all_days_morning_mean.append(this_day)
    all_days_afternoon_mean = []
    for j in range(len(all_data[afternoon])):
        temp = all_data[afternoon][j]
        for k in range(30):
            this_day = np.mean(temp[k*60:(k+1)*60])
            all_days_afternoon_mean.append(this_day)
    all_days_night_mean = []
    for j in range(len(all_data[night])):
        temp = all_data[night][j]
        for k in range(30):
            this_day = np.mean(temp[k*60:(k+1)*60])
            all_days_night_mean.append(this_day)

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure()
    plt.hist(all_days_morning_mean, bins=10, color='black', rwidth=0.9, orientation='horizontal') #lightcoral
    plt.title("每日早7时风速均值统计直方图")
    plt.savefig("每日早7时风速均值统计直方图.png")
    plt.close()

    plt.figure()
    plt.hist(all_days_afternoon_mean, bins=10, color='black', rwidth=0.9, orientation='horizontal') #springgreen
    plt.title("每日午14时风速均值统计直方图")
    plt.savefig("每日午14时风速均值统计直方图.png")
    plt.close()

    plt.figure()
    plt.hist(all_days_night_mean, bins=10, color='black', rwidth=0.9, orientation='horizontal') #blueviolet
    plt.title("每日晚21时风速均值统计直方图")
    plt.savefig("每日晚21时风速均值统计直方图.png")
    plt.close()


    four_wind = [0, 10, 20, 30, round(max_wind)]
    print(four_wind)
    sixtheen_wind = ['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW']
    a = np.zeros((4,16))
    for i in range(len(all_data)):
        for j in range(len(all_data[i])):
            temp_wind = all_data[i][j]
            temp_wind2 = all_data_wind2[i][j]
            x_index = 0
            y_index = 0
            for k in range(len(temp)):
                if four_wind[0]<=temp[k]<four_wind[1]:
                    x_index = 0
                if four_wind[1]<=temp[k]<four_wind[2]:
                    x_index = 1
                if four_wind[2]<=temp[k]<four_wind[3]:
                    x_index = 2
                if four_wind[3]<=temp[k]<four_wind[4]:
                    x_index = 3
                y_index = sixtheen_wind.index(all_data_wind2[i][j][k])
                a[x_index,y_index] += 1
    data = pd.DataFrame(a,
                        index=['0~'+str(four_wind[1]), str(four_wind[1])+'~'+str(four_wind[2]), str(four_wind[2])+'~'+str(four_wind[3]), str(four_wind[3])+'~'+str(four_wind[4])],
                        columns='N NNE NE ENE E ESE SE SSE S SSW SW WSW W WNW NW NNW'.split())
    N = 16  # 风速分布为16个方向
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)  # 获取16个方向的角度值
    width = np.pi / N  # 绘制扇型的宽度，可以自行调整
    labels = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW','NNW']
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, projection='polar')
    for idx in data.index:
        radii = data.loc[idx]  # 每一行数据
        ax.bar(theta, radii, width=width, bottom=0.0, label=idx, tick_label=labels)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    plt.title('风玫瑰图示意图')
    plt.legend(loc=4, bbox_to_anchor=(1.15, -0.07))
    plt.savefig("风向风速玫瑰图.png")
    plt.show()
    return all_data_new2

def get_self_data(path):
    wind_dire = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    data = xlrd.open_workbook(path)
    table = data.sheets()[0]
    nrows = table.nrows
    temp_len = (nrows-1)/3
    count = 0
    temp_data = []
    my_all_data = []

    temp_wind = []
    my_all_data_wind = []

    temp_wind2 = []
    all_data_wind2 = []


    for i in range(1,nrows):
        if i!=0 and i%temp_len==0:
            temp_data.append(float(table.row_values(i)[1].split('m/s')[0]))
            my_all_data.append([temp_data])
            temp_data = []

            temp_wind.append( int(table.row_values(i)[2].split('°')[0].strip(' ')) )
            my_all_data_wind.append([temp_wind])
            temp_wind = []

            temp_wind2.append( wind_dire[math.floor(int(table.row_values(i)[2].split('°')[0].strip(' ')) / 22.5)] )
            all_data_wind2.append([temp_wind2])
            temp_wind2 = []
        else:
            temp_data.append(float(table.row_values(i)[1].split('m/s')[0]))
            temp_wind.append(int(table.row_values(i)[2].split('°')[0].strip(' ')))
            temp_wind2.append(wind_dire[math.floor(int(table.row_values(i)[2].split('°')[0].strip(' ')) / 22.5)])
    return my_all_data, my_all_data_wind, all_data_wind2





    # my_all_data_new = copy.deepcopy(my_all_data)
    # for i in range(len(my_all_data)): #i遍历早中晚
    #     for j in range(len(my_all_data[i])): #j遍历多少个月 咱们这里就是一个月的情况
    #         temp = my_all_data[i][j] #每天10个值 30天共300个值
    #         temp_new = []
    #         temp_wind = []
    #         temp_wind_cor = []
    #         for k in range(30): #k
    #             x = np.arange(10)*6
    #             y = np.array(temp[k*10:(k+1)*10])
    #             #3次高阶拟合
    #             f1 = np.polyfit(x, y, 3)
    #             x_new = np.arange(60)
    #             y_new = np.polyval(f1, x_new)
    #             for l in range(y_new.shape[0]):
    #                 temp_new.append(abs(y_new[l]))
    #         my_all_data_new[i][j] = temp_new

if __name__ == '__main__':
    all_data = get_data(month=7)
    #先计算早中晚风速的均值和方差
    mean, std = analysis_1(all_data)
    #采用多项式拟合 对数据进行挖掘补全 原本10分钟间隔的数据 变为1分钟间隔
    all_data_new, all_data_wind, all_data_wind2 = analysis_2(all_data)
    all_data_new2 = analysis_3(all_data_new, all_data_wind, all_data_wind2)

    # 如果是已有的数据 只需要保存在test.xls里 运行下面的就行
    # all_data_new, all_data_wind, all_data_wind2= get_self_data('./test.xls')
    # mean, std = analysis_1(all_data)
    # all_data_new2 = analysis_3(all_data_new, all_data_wind, all_data_wind2)
