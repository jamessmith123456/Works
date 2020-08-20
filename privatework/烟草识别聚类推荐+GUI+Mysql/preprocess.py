'''
主题：烟叶评价
设计一种面向烟叶评价业务并融合多种评价方法的软件系统，各种评价方法的结果对于
人工评价而言具有一定的参考性，降低人工评价阶段的工作量，与人工评价结果对比时
可有效减小人的主观偏差导致的误差。

主要功能：
1.评价数据的IO功能
2.感官指标综合评价
3.烟叶化学成分指标综合评价
4.烟叶化学成分对烟叶感官得分的预测


感官指标综合评价：
1.感官指标被分为两大类，品质特征和风格特征    品质特征包括：香气量、浓度、刺激性、劲头等；  风格特征包括：各个香气特征
主成分分析法进行分析   分析之后得到第一主成分、第二主成分、第三主成分... (在PCA之前需要进行数据中心化处理 即 （原始数据-指标期望)/(指标标准差) )

化学成分综合评价：
1.绝对评价，即通过指标与最优值的比较说明指标质量的优劣
模糊综合评价法   灰色综合评价法

感官得分预测：
人工神经网络


多指标综合评价（统计综合评价）：对事物或现象的多个侧面量化统计成多个评价指标并结合一定的数学工具将指标进行综合分析比较；
1.指标筛选  2.无量纲化 3.指标赋权  4.指标合成

在绝对评价领域需要对指标的绝对好坏进行评判，一般用在无量纲化的方法时，每个指标都有其最优值，根据指标需求类型将指标分为三种类型；
效益型指标:指标值越大说明指标越趋近最优值  如某地GDP 在系统中对应的键值为大
成本型指标:指标值越小说明指标越接近最优值  如烟草的有害物质 系统中对应的键值为小
适中型指标:指标拥有最佳值或最佳值域区间 越接近说明越趋近于最优值 系统中对应的键值为适中


四份数据集中：
./data/2015-16年样品检测结果汇总表--邱昌桂（0606）.xlsx这个excel的常规化学、有机酸、多酚、生物碱、水溶性糖、致香
成分、色素这几个表都包含在./data/2015年感官质量.xlsx这个表里 物理指标这个表是没有的，但是它也不属于化学指标，所以没有也没事



2010-2013年云南、广东、湖南等各地419的样本的感官指标集合
2015年云南C3F的包含111个化学指标的91个样本集合
'''

import numpy as np
import xlrd
from sklearn.decomposition import PCA
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import copy
import pymysql

def get_data_1(path):
    data = xlrd.open_workbook(path)
    table = data.sheet_by_index(0)
    print("table-name:{},table-rows:{},table-cols:{}".format(table.name,table.nrows,table.ncols))

    example_index = 1
    chemistry_index= list(range(8,119))
    human_index = list(range(119,127))
    score_index = 127

    chemistry_name = []
    for item in chemistry_index:
        chemistry_name.append(table.cell(0,item).value)
    human_name = []
    for item in human_index:
        human_name.append(table.cell(0,item).value)

    all_example_feature = []
    for i in range(1,table.nrows):
        try:
            temp_example_all_feature = {}
            temp_example_name = table.cell(i,example_index).value
            temp_example_chemistry_feature = []
            for item in chemistry_index:
                try:
                    temp_example_chemistry_feature.append(float(table.cell(i,item).value))
                except:
                    temp_example_chemistry_feature.append(float('nan'))
            temp_example_human_feature = []
            for item in human_index:
                try:
                    temp_example_human_feature.append(float(table.cell(i,item).value))
                except:
                    temp_example_human_feature.append(float('nan'))
            temp_example_score = float(table.cell(i,score_index).value)

            temp_example_all_feature['example_name'] = temp_example_name
            temp_example_all_feature['example_chemistry'] = temp_example_chemistry_feature
            temp_example_all_feature['example_human'] = temp_example_human_feature
            temp_example_all_feature['example_score'] = temp_example_score

            all_example_feature.append(temp_example_all_feature)
        except:
            print(i)

    all_example_chemistry = []
    for i in range(len(all_example_feature)):
        all_example_chemistry.append(all_example_feature[i]['example_chemistry'])

    all_example_human = []
    for i in range(len(all_example_feature)):
        all_example_human.append(all_example_feature[i]['example_human'])

    all_example_score = []
    for i in range(len(all_example_feature)):
        all_example_score.append(all_example_feature[i]['example_score'])

    return all_example_feature, chemistry_name, human_name, np.array(all_example_chemistry), np.array(all_example_human), np.array(all_example_score)

def get_data_2(path):
    data = xlrd.open_workbook(path)
    table = data.sheet_by_index(0)
    print("table-name:{},table-rows:{},table-cols:{}".format(table.name,table.nrows,table.ncols))

    example_index = 1
    chemistry_index= list(range(8,111))

    chemistry_name = []
    for item in chemistry_index:
        chemistry_name.append(table.cell(0,item).value)

    all_example_feature = []
    for i in range(1,table.nrows):
        try:
            temp_example_all_feature = {}
            temp_example_name = table.cell(i,example_index).value
            temp_example_chemistry_feature = []
            for item in chemistry_index:
                try:
                    temp_example_chemistry_feature.append(float(table.cell(i,item).value))
                except:
                    temp_example_chemistry_feature.append(float('nan'))

            temp_example_all_feature['example_name'] = temp_example_name
            temp_example_all_feature['example_chemistry'] = temp_example_chemistry_feature

            all_example_feature.append(temp_example_all_feature)
        except:
            print(i)

    all_example_chemistry = []
    for i in range(len(all_example_feature)):
        all_example_chemistry.append(all_example_feature[i]['example_chemistry'])

    return all_example_feature, chemistry_name, np.array(all_example_chemistry)

def remove_nan(a):
    a_mean = np.nanmean(a,0)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if math.isnan(a[i,j]):
                a[i,j] = a_mean[j]
    return a

def pca_analysis(all_example_human):
    #主成分分析法，基于数据集内部的方差对样本进行感官得分的评价，并完成对样本的排序，用来解决烟叶感官质量各指标对综合得分贡献度难以解决的问题
    pca = PCA(n_components=all_example_human.shape[1], copy=True, whiten=True)
    new_all_example_human = pca.fit_transform(all_example_human)
    #pca现在有如下几个属性
    print('pca.n_components:', pca.n_components)#返回所保留的成分个数n
    print('pca.explained_variance_ratio_:', pca.explained_variance_ratio_)#返回 所保留的n个成分各自的方差百分比 也即贡献率
    print('pca.components_:', pca.components_)#返回具有最大方差的成分
    print('pca.explained_variance_:', pca.explained_variance_ )#返回 所保留的n个成分各自的方差  方差小说明没啥作用，可以忽略 保留方差大的

    #针对新的数据a 可以利用训练好的pca进行处理 newa = pca.transform(a)

    return new_all_example_human, pca.components_, pca.explained_variance_ratio_

def fuzzy_comprehensive_evaluation(all_example_chemistry, all_example_score, example):
    #采用模糊综合评判法对烟叶的化学成分指标进行得分统计 根据样本值与最佳值之间的隶属度评价结果来描述烟叶样本质量在烟草质量体系中的绝对位置
    all_col_max = np.max(all_example_chemistry, 0) #(111,) 注意传进来的参数不能有nan值 否则对样本的最大最小值判断会有影响
    all_col_min = np.min(all_example_chemistry, 0) #(111,)
    #这边因为指标数目很多 所以直接对指标样本进行相似性评估 通过计算余弦距离的方式，来评判某个样本在整体质量体系中的绝对位置

    max_score_index = np.argmax(all_example_score)
    max_score = np.max(all_example_score)

    min_score_index = np.argmin(all_example_score)
    min_score = np.min(all_example_score)

    dis1 = np.dot(example, all_example_chemistry[max_score_index,:]) / (np.linalg.norm(example) * (np.linalg.norm(all_example_chemistry[max_score_index,:])))
    dis2 = np.dot(all_example_chemistry[min_score_index,:], all_example_chemistry[max_score_index,:]) / (np.linalg.norm(all_example_chemistry[min_score_index,:]) * (np.linalg.norm(all_example_chemistry[max_score_index,:])))

    example_score = abs(dis1/dis2)*(max_score - min_score) + min_score
    return example_score

def network(x_train, y_train, x_test):
    # 定义两个占位符
    x = tf.placeholder(tf.float32, [None, x_train.shape[1]])  # 形状为n行1列，同x_data的shape
    y = tf.placeholder(tf.float32, [None])
    # 定义神经网络
    # 定义中间层，因为每个x是一维，所以只需1个神经元，定义中间层的连接神经元是10
    # 矩阵：[a, b]×[b, c] = [a, c]
    L1_weights = tf.Variable(tf.random_normal([x_train.shape[1], x_train.shape[1]*2]))
    L1_bias = tf.Variable(tf.zeros([1, x_train.shape[1]*2]))
    L1_weights_bias = tf.matmul(x, L1_weights) + L1_bias
    L1 = tf.nn.sigmoid(L1_weights_bias)
    #L1 = L1_weights_bias
    # 定义输出层，每个x只有一个神经元
    L2_weights = tf.Variable(tf.random_normal([x_train.shape[1]*2, 1]))
    L2_bias = tf.Variable(tf.zeros([1, 1]))
    L2_weights_bias = tf.matmul(L1, L2_weights) + L2_bias
    L2 = L2_weights_bias
    # 定义损失函数
    loss = tf.reduce_mean(tf.square(y - L2))
    # 梯度下降最小化损失函数
    optimizer = tf.train.GradientDescentOptimizer(0.001)
    train_step = optimizer.minimize(loss)
    # 全局变量初始化
    init = tf.global_variables_initializer()
    # 定义会话
    with tf.Session() as sess:
        sess.run(init)
        for _ in range(20):
            sess.run(train_step, feed_dict={x: x_train, y: y_train})
            #print('loss:',sess.run(loss, feed_dict={x: x_train, y: y_train}))
        # 获取预测值
        predict = sess.run(L2, feed_dict={x: x_test})
    return predict

def create_mysql(chemistry_name, human_name, chemistry_name2):
    db = pymysql.connect("localhost", "root", "123456", "smoke")
    # 远程访问指定IP上的数据库

    # 使用 cursor() 方法创建一个游标对象 cursor
    cursor = db.cursor()

    sql_createTb = """CREATE TABLE chemistry_feature (
                      id CHAR(80) NOT NULL,
                      """
    for i in range(len(chemistry_name)):
        temp1 = chemistry_name[i].replace(' ', '')
        temp1 = temp1.replace('-', '')
        temp1 = temp1.replace(',', '')
        temp1 = temp1.replace('(', '')
        temp1 = temp1.replace(')', '')
        temp1 = temp1.replace('+', '')
        temp1 = temp1.replace('[', '')
        temp1 = temp1.replace(']', '')
        temp1 = temp1.replace('\'', '')
        temp = "                  "+temp1 + " " +"DOUBLE(7,3)," +"\n"
        sql_createTb += temp
    sql_createTb = sql_createTb[:-2]+")"
    #sql_createTb = sql_createTb.replace('β-','')
    #sql_createTb = sql_createTb.replace('-', '')
    #sql_createTb = sql_createTb.replace(',', '')
    sql_createTb.encode('utf-8')
    cursor.execute(sql_createTb)  # 只用创建一次，再次执行会出错



    sql_createTb = """CREATE TABLE human_feature (
                      id CHAR(80) NOT NULL,
                      """
    for i in range(len(human_name)):
        temp1 = human_name[i].replace(' ', '')
        temp1 = temp1.replace('-', '')
        temp1 = temp1.replace(',', '')
        temp1 = temp1.replace('(', '')
        temp1 = temp1.replace(')', '')
        temp1 = temp1.replace('+', '')
        temp1 = temp1.replace('[', '')
        temp1 = temp1.replace(']', '')
        temp1 = temp1.replace('\'', '')
        temp = "                  "+temp1 + " " +"DOUBLE(7,3)," +"\n"
        sql_createTb += temp
    sql_createTb = sql_createTb[:-2]+")"
    sql_createTb.encode('utf-8')
    cursor.execute(sql_createTb)  # 只用创建一次，再次执行会出错

    sql_createTb = """CREATE TABLE chemistry_feature2 (
                      id CHAR(80) NOT NULL,
                      """
    chemistry_name2 = list(set(chemistry_name2))
    for i in range(len(chemistry_name2)):
        temp1 = chemistry_name2[i].replace(' ', '')
        temp1 = temp1.replace('-', '')
        temp1 = temp1.replace(',', '')
        temp1 = temp1.replace('(', '')
        temp1 = temp1.replace(')', '')
        temp1 = temp1.replace('+', '')
        temp1 = temp1.replace('[', '')
        temp1 = temp1.replace(']', '')
        temp1 = temp1.replace('\'', '')
        temp = "                  "+temp1 + " " +"DOUBLE(7,3)," +"\n"
        sql_createTb += temp
    sql_createTb = sql_createTb[:-2]+")"
    sql_createTb.encode('utf-8')
    cursor.execute(sql_createTb)  # 只用创建一次，再次执行会出错

def write_to_mysql(feature_id, feature, name, table):
    db = pymysql.connect("localhost", "root", "123456", "smoke")
    cursor = db.cursor()
    for i in range(len(name)):
        name[i] = name[i].replace(' ', '')
        name[i] = name[i].replace('-', '')
        name[i] = name[i].replace(',', '')
        name[i] = name[i].replace('(', '')
        name[i] = name[i].replace(')', '')
        name[i] = name[i].replace('+', '')
        name[i] = name[i].replace('[', '')
        name[i] = name[i].replace(']', '')
        name[i] = name[i].replace('\'', '')

    for i in range(len(feature_id)):
        sql_createTb = """INSERT INTO """+ table +"(id, "
        sql_createTb += (','.join(name))
        sql_createTb += ') VALUES (\''+ feature_id[i] +'\','
        temp_value = [str(feature[i,j]) for j in range(feature.shape[1])]
        sql_createTb += (','.join(temp_value))
        sql_createTb += ')'
        try:
            cursor.execute(sql_createTb)
            db.commit()
        except:
            db.rollback()
    db.close()

def read_from_mysql(table):
    db = pymysql.connect("localhost", "root", "123456", "smoke")
    cursor = db.cursor()
    sql_createTb = 'SELECT * FROM '+ table
    all_data_num = cursor.execute(sql_createTb)
    all_data = cursor.fetchall()

    feature_id = []
    feature = []
    for i in range(len(all_data)):
        feature_id.append(all_data[i][0])
        feature.append(list(all_data[i][1:]))
    feature = np.array(feature)
    return feature_id , feature

if __name__ == '__main__':
    path = './data/2015年感官质量.xlsx'
    all_example_feature, chemistry_name, human_name, all_example_chemistry, all_example_human, all_example_score = get_data_1(path)
    all_example_chemistry = remove_nan(all_example_chemistry) #(91,111)
    all_example_human = remove_nan(all_example_human) #(91,8)  #all_example_score的shape是(91,)

    path = './data/卷烟检测结果.xlsx'
    all_example_feature2, chemistry_name2, all_example_chemistry2 = get_data_2(path)
    all_example_chemistry2 = remove_nan(all_example_chemistry2) #(43,103)

    #创建数据库 并且将数据写入数据库
    create_mysql(chemistry_name, human_name, chemistry_name2)
    feature_id = [item['example_name'] for item in all_example_feature]
    write_to_mysql(feature_id, all_example_chemistry , chemistry_name, 'chemistry_feature')
    write_to_mysql(feature_id, all_example_human, human_name, 'human_feature')
    feature_id2 = [item['example_name'] for item in all_example_feature2]
    write_to_mysql(feature_id2, all_example_chemistry2, chemistry_name2, 'chemistry_feature2')

    #重新从数据库中读取数据
    feature_id1, feature1 = read_from_mysql('chemistry_feature')
    feature_id2, feature2 = read_from_mysql('human_feature')
    feature_id3, feature3 = read_from_mysql('chemistry_feature2')

    #主成分分析
    new_all_example_human, pca_components_, pca_explained_variance_ratio_ = pca_analysis(all_example_human)

    #模糊综合评价法
    score = fuzzy_comprehensive_evaluation(all_example_chemistry, all_example_score, all_example_chemistry[0,:])
    print('score:', score)

    #神经网络
    x_train = copy.deepcopy(all_example_chemistry)
    y_train = copy.deepcopy(all_example_score)
    x_test = copy.deepcopy(all_example_chemistry[0,:])
    x_test = x_test[np.newaxis,:]
    predict = network(x_train, y_train, x_test)
    print('predict:', predict)