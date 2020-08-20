#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import pickle
import pandas as pd
plt.switch_backend('agg') #去掉图形界面
import time
import math
import copy
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

def normalize(X, means1=None, std1=None, means2=None, std2=None, norm='tanh_norm'):
    if std1 is None:
        std1 = np.nanstd(X, axis=0)
    if means1 is None:
        means1 = np.mean(X, axis=0)
    X = np.ascontiguousarray(X)
    if norm is None:
        return (X, means1, std1)
    if norm == 'norm':
        X = (X - means1) / std1
        return(X, means1, std1)
    elif norm == 'tanh':
        return(np.tanh(X), means1, std1)
    elif norm == 'tanh_norm':
        X = np.tanh(X)
        if means2 is None:
            means2 = np.mean(X, axis=0)
        if std2 is None:
            std2 = np.std(X, axis=0)
        X = (X-means2)/std2
        return(X, means1, std1, means2, std2)

def binary(data, threshold = 30):
    data[ data < threshold ] = 0
    data[ data >= threshold ] = 1
    return data

def get_data(path1, path2, path3, val_fold=0,norm='tanh'):
    file = open(path1, 'rb')
    X = pickle.load(file)
    file.close()

    file = open(path2, 'rb')
    X_miss = pickle.load(file)
    file.close()

    data = pd.read_csv(path3)
    double_data = pd.concat([data, data])

    idx_train = np.where(double_data['fold']!=val_fold)
    idx_val = np.where(double_data['fold']==val_fold)

    X_train = X[idx_train]
    X_val = X[idx_val]

    y_train = double_data.iloc[idx_train]['synergy'].values
    y_train = binary(y_train)
    y_val = double_data.iloc[idx_val]['synergy'].values
    y_val = binary(y_val)

    if norm == "tanh_norm":
        X_train, mean, std, mean2, std2 = normalize(X_train, norm=norm)
        X_val, mean, std, mean2, std2 = normalize(X_val, mean, std, mean2, std2,  norm=norm)
        X_miss, mean, std, mean2, std2 = normalize(X_miss, mean, std, mean2, std2, norm=norm)
    else:
        X_train, mean, std = normalize(X_train, norm=norm)
        X_val, mean, std = normalize(X_val, mean, std, norm=norm)
        X_miss, mean, std = normalize(X_miss, mean, std, norm=norm)

    y_train_one_hot = np.zeros((y_train.shape[0],2))
    y_val_one_hot = np.zeros((y_val.shape[0], 2))
    for i in range(y_train.shape[0]):
        y_train_one_hot[i, int(y_train[i])] = 1
    for i in range(y_val.shape[0]):
        y_val_one_hot[i, int(y_val[i])] = 1

    print("fold:",val_fold)
    print("X_train:", X_train.shape)
    print("y_train:", y_train_one_hot.shape)
    print("X_val:", X_val.shape)
    print("y_val:", y_val_one_hot.shape)
    return X_train, y_train_one_hot, X_val, y_val_one_hot, X_miss

def get_balanced_sub_data(X_train, y_train):
    pos_index = []
    neg_index = []
    for i in range(X_train.shape[0]):
        if y_train[i,0]==1:
            neg_index.append(i)
        else:
            pos_index.append(i)
    print("before subsampling, pos_num:{},neg_num:{}".format(len(pos_index),len(neg_index)))
    np.random.shuffle(neg_index)
    pos_index.extend(neg_index[:len(pos_index)])

    balanced_X_train = X_train[pos_index,:]
    balanced_y_train = y_train[pos_index,:]
    pos_index = []
    neg_index = []
    for i in range(balanced_X_train.shape[0]):
        if balanced_y_train[i,0]==1:
            neg_index.append(i)
        else:
            pos_index.append(i)
    print("after subsampling, pos_num:{},neg_num:{}".format(len(pos_index),len(neg_index)))

    return balanced_X_train, balanced_y_train

def get_balanced_sub_data2(X_train, y_train):
    pos_index = []
    neg_index = []
    for i in range(X_train.shape[0]):
        if y_train[i,0]==1:
            neg_index.append(i)
        else:
            pos_index.append(i)
    print("before subsampling, pos_num:{},neg_num:{}".format(len(pos_index),len(neg_index)))
    np.random.shuffle(neg_index)
    sub_num = math.floor(len(neg_index)/len(pos_index))

    all_bal_train_test = []
    for i in range(sub_num):
        pos_index_copy = copy.deepcopy(pos_index)
        pos_index_copy.extend(neg_index[i*len(pos_index):(i+1)*len(pos_index)])

        balanced_X_train = X_train[pos_index_copy,:]
        balanced_y_train = y_train[pos_index_copy,:]
        pos_index_aft = []
        neg_index_aft = []
        for i in range(balanced_X_train.shape[0]):
            if balanced_y_train[i,0]==1:
                neg_index_aft.append(i)
            else:
                pos_index_aft.append(i)
        print("after subsampling, pos_num:{},neg_num:{}".format(len(pos_index_aft),len(neg_index_aft)))
        all_bal_train_test.append([balanced_X_train, balanced_y_train])
    return all_bal_train_test

def my_next_batch(X_train, y_train, batch_size):
    # 打乱数据集
    index = [i for i in range(0, X_train.shape[0])]
    np.random.shuffle(index)
    # 建立batch_data与batch_target的空列表
    batch_data = []
    batch_target = []
    # 向空列表加入训练集及标签
    for i in range(0, batch_size):
        batch_data.append(list(X_train[index[i],:]))
        batch_target.append(y_train[index[i]])
    batch_data = np.array(batch_data)
    batch_target = np.array(batch_target)
    return batch_data, batch_target  # 返回

def my_next_batch2(X_train, y_train, batch_size, batch_num):
    #这种batchsize采样方式，需要实现先把数据打乱
    start_index = (batch_num*batch_size)%X_train.shape[0]
    if (X_train.shape[0]-start_index)<=batch_size:
        end_index = X_train.shape[0]
    else:
        end_index = ((batch_num+1)*batch_size)%X_train.shape[0]
    batch_data = X_train[start_index:end_index,:]
    batch_target = y_train[start_index:end_index,:]
    return batch_data, batch_target  # 返回


def CNN_network(X_train, y_train, X_val, y_val, param1, param2, name1, name2):
    with tf.name_scope('input'):
        global_step = tf.Variable(0)
        input_x = tf.placeholder(tf.float32, [None, 3000])
        output_y = tf.placeholder(tf.float32, [None, 2])
        input_x_images = tf.reshape(input_x, [-1, 1, 3000, 1])

    with tf.name_scope('CNN'):
        conv1=tf.layers.conv2d(
            inputs=input_x_images, #(None,1,3000,1)
            filters=16, #卷积核个数
            kernel_size=[1,5], #卷积核尺寸
            strides=1, #扫描步长
            padding='same', #边边补0 valid不需要补0，same需要补0，为了保证输入输出的尺寸一致,补多少不需要知道
            activation=tf.nn.relu #激活函数
        )
        pool1 = tf.layers.max_pooling2d(
            inputs=conv1, #(None,1,3000,1)
            pool_size=[1, 2], #过滤器的尺寸
            strides=2
        )
        conv2 = tf.layers.conv2d(
            inputs=pool1, #(None,1,1500,8)
            filters=16,
            kernel_size=[1, 5],
            strides=1,
            padding='same',
            activation=tf.nn.relu
        )
        pool2 = tf.layers.max_pooling2d(
            inputs=conv2, #(None,1,1500,8)
            pool_size=[1, 2],
            strides=2
        )
        conv3 = tf.layers.conv2d(
            inputs=pool2, #(None,1,750,8)
            filters=32,
            kernel_size=[1, 5],
            strides=1,
            padding='same',
            activation=tf.nn.relu
        )
        pool3 = tf.layers.max_pooling2d(
            inputs=conv3, #(None,1,750,16)
            pool_size=[1, 2],
            strides=2
        )
        conv4 = tf.layers.conv2d(
            inputs=pool3, #(None,1,375,16)
            filters=32,
            kernel_size=[1, 5],
            strides=1,
            padding='same',
            activation=tf.nn.relu
        )
        pool4 = tf.layers.max_pooling2d(
            inputs=conv4, #(None,1,375,16)
            pool_size=[1, 2],
            strides=2
        )
        flat = tf.reshape(pool4, [-1, 1 * 187 * 32])
        dense = tf.layers.dense(
            inputs=flat,
            units=2500,
            activation=tf.nn.relu
        )
        dense1 = tf.layers.dense(
            inputs=dense,
            units=1024,
            activation=tf.nn.relu
        )
        dense2 = tf.layers.dense(
            inputs=dense1,
            units=1024,
            activation=tf.nn.relu
        )
        dense3 = tf.layers.dense(
            inputs=dense2,
            units=256,
            activation=tf.nn.relu
        )
        # with tf.name_scope('dropout'):
        #     dropout = tf.layers.dropout(
        #         inputs=dense,
        #         rate=0.2,
        #     )
        logits = tf.layers.dense(
            inputs=dense3,
            units=2
        )
    # loss = tf.losses.softmax_cross_entropy(onehot_labels=output_y,
    #                                        logits=logits)
    cost1 = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=output_y)
    alpha = tf.constant([[param1, param2]])
    cost2 = cost1*alpha
    loss = tf.reduce_mean(cost1)
    # 用Adam 优化器来最小化误差,学习率0.001 类似梯度下降
    initial_learning_rate = tf.Variable(0.001)

    learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                               global_step=global_step,
                                               decay_steps=300000, #总共训练2万次，每6千次减少为一半
                                               decay_rate=0.5)
    train_op = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)
    accuracy_op=tf.metrics.accuracy(
        labels=tf.argmax(output_y,axis=1),
        predictions=tf.argmax(logits,axis=1)
    )[1]

    init = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
    with tf.Session() as sess:
        loss_array = []
        test_ratio = []
        sess.run(init)
        writer = tf.summary.FileWriter('logs/', sess.graph) #tf.get_default_graph()

        index = [i for i in range(0, X_train.shape[0])]
        np.random.shuffle(index)
        X_train_shuffle = X_train[index,:]
        y_train_shuffle = y_train[index,:]
        saver = tf.train.Saver(max_to_keep=1)
        for i in range(300000):
            #batch_xs, batch_ys = my_next_batch(X_train, y_train, 16)  # 设置每次训练样本数batch
            batch_xs, batch_ys = my_next_batch2(X_train_shuffle, y_train_shuffle, 16, i)
            train_loss, test_accuracy, _, train_learning_rate = sess.run(
                    [loss, accuracy_op, train_op, learning_rate],
                    feed_dict = {input_x:batch_xs ,output_y:batch_ys ,global_step:i})
            # if i%50000==0:
            #     #每5万次保存一个模型
            #     print("iter:{}, learning_rate:{}".format(i, train_learning_rate))
            #     saver.save(sess, "./saved_model", global_step=i) #保存当前已经训练的模型 留到后面训练
            #
            #     #每5万次保存损失图片并且计算准确率
            #     print("<<<<<<<<<<<<<<<<<<<<<<<<")
            #     fig = plt.figure()  # 创建图片
            #     ax1 = fig.add_subplot(1, 2, 1)  # 显示损失值
            #     ax2 = fig.add_subplot(1, 2, 2)  # 显示测试集准确率
            #     # ------损失值图形--------
            #     ax1.plot(loss_array, "r--")
            #     ax1.set_title("loss value")
            #     # ------测试集准确率图形--------
            #     ax2.plot(test_ratio, "b--")
            #     ax2.set_title("test accuary")
            #     plt.savefig('cnn_train' + str(name1) + '_' + str(name2) + '.png')
            #
            #     test_output = sess.run(logits, {input_x: X_val})
            #     inferenced_y = np.argmax(test_output, 1)
            #
            #     TP = 0
            #     TN = 0
            #     FP = 0
            #     FN = 0
            #     for i in range(inferenced_y.shape[0]):
            #         if inferenced_y[i] == 1 and y_val[i, 1] == 1:
            #             TP += 1
            #         if inferenced_y[i] == 0 and y_val[i, 0] == 1:
            #             TN += 1
            #         if inferenced_y[i] == 1 and y_val[i, 0] == 1:
            #             FP += 1
            #         if inferenced_y[i] == 0 and y_val[i, 1] == 1:
            #             FN += 1
            #
            #     print('val_num:', y_val.shape[0])
            #     print('right_num:', TP + TN)
            #     Accuracy = (TP + TN) / y_val.shape[0]
            #     Precision = TP / (TP + FP) if (TP + FP) != 0 else -1.234
            #     Recall = TP / (TP + FN) if (TP + FN) != 0 else -1.234
            #     p0 = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else -1.234
            #     pe = ((TP + FP) * (TP + FN) + (FP + TN) * (FN + TN)) / ((TP + TN + FP + FN) * (TP + TN + FP + FN))
            #     Kappa = (p0 - pe) / (1 - pe)
            #     print("TP:{}, TN:{}, FP:{}, FN:{}, Accuracy:{}, Precision:{}, Recall:{}, Kappa:{}".format(TP, TN, FP, FN,
            #                                                                                               Accuracy,
            #                                                                                               Precision, Recall,
            #                                                                                               Kappa))
            loss_array.append(train_loss)
            test_ratio.append(test_accuracy)
        writer.close()
        test_output=sess.run(logits,{input_x:X_val})
        inferenced_y=np.argmax(test_output,1)

        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for i in range(inferenced_y.shape[0]):
            if inferenced_y[i] == 1 and y_val[i, 1] == 1:
                TP += 1
            if inferenced_y[i] == 0 and y_val[i, 0] == 1:
                TN += 1
            if inferenced_y[i] == 1 and y_val[i, 0] == 1:
                FP += 1
            if inferenced_y[i] == 0 and y_val[i, 1] == 1:
                FN += 1

        print('val_num:', y_val.shape[0])
        print('right_num:', TP + TN)
        Accuracy = (TP + TN) / y_val.shape[0]
        Precision = TP / (TP + FP) if (TP + FP) != 0 else -1.234
        Recall = TP / (TP + FN) if (TP + FN) != 0 else -1.234
        p0 = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else -1.234
        pe = ((TP + FP) * (TP + FN) + (FP + TN) * (FN + TN)) / ((TP + TN + FP + FN) * (TP + TN + FP + FN))
        Kappa = (p0 - pe) / (1 - pe)
        print("TP:{}, TN:{}, FP:{}, FN:{}, Accuracy:{}, Precision:{}, Recall:{}, Kappa:{}".format(TP, TN, FP, FN, Accuracy,
                                                                                                  Precision, Recall, Kappa))
        print("*********************")
        fig = plt.figure()  # 创建图片
        ax1 = fig.add_subplot(1, 2, 1)  # 显示损失值
        ax2 = fig.add_subplot(1, 2, 2)  # 显示测试集准确率
        # ------损失值图形--------
        ax1.plot(loss_array, "r--")
        ax1.set_title("loss value")
        # ------测试集准确率图形--------
        ax2.plot(test_ratio, "b--")
        ax2.set_title("test accuary")
        plt.savefig('cnn_train'+str(name1)+'_'+str(name2)+'.png')

        return Accuracy, inferenced_y, test_output, TP, TN, FP, FN

def bagging(all_bal_train_test, X_val, y_val, param1, param2, name):
    all_ratio = []
    all_y = []
    all_test_output = []
    for i in range(len(all_bal_train_test)):
        temp_bal_X_train = all_bal_train_test[i][0]
        temp_bal_y_train = all_bal_train_test[i][1]
        tempacc, tempy, temptest_output, temp_TP, temp_TN, temp_FP, temp_FN = CNN_network(temp_bal_X_train, temp_bal_y_train, X_val, y_val, param1, param2, name ,i)
        all_ratio.append(tempacc)
        all_y.append(tempy)
        all_test_output.append(temptest_output)
    save_file = open("all_ratio_y"+ name +".pkl","wb")
    pickle.dump(all_ratio, save_file)
    pickle.dump(all_y, save_file)
    pickle.dump(all_test_output, save_file)
    pickle.dump(y_val, save_file)
    save_file.close()

    all_ratio_new = [all_ratio[i]/ sum(all_ratio) for i in range(len(all_ratio))]
    all_result = []
    for i in range(len(all_y[0])):
        all_result.append([0,0])
    for i in range(len(all_ratio_new)):
        temp_ratio = all_ratio_new[i]
        temp_y = all_y[i]
        for j in range(len(temp_y)):
            all_result[j][temp_y[j]] += temp_ratio

    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(y_val.shape[0]):
        if all_result[i][0] > all_result[i][1] and y_val[i, 0] == 1:
            TN += 1
        if all_result[i][0] > all_result[i][1] and y_val[i, 1] == 1:
            FN += 1
        if all_result[i][0] < all_result[i][1] and y_val[i, 0] == 1:
            FP += 1
        if all_result[i][0] < all_result[i][1] and y_val[i, 1] == 1:
            TP += 1
    Accuracy = (TP + TN) / y_val.shape[0]
    Precision = TP / (TP + FP) if (TP + FP) != 0 else -1.234
    Recall = TP / (TP + FN) if (TP + FN) != 0 else -1.234
    p0 = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else -1.234
    pe = ((TP + FP) * (TP + FN) + (FP + TN) * (FN + TN)) / ((TP + TN + FP + FN) * (TP + TN + FP + FN))
    Kappa = (p0 - pe) / (1 - pe)
    print("TP:{}, TN:{}, FP:{}, FN:{}, Accuracy:{}, Precision:{}, Recall:{}, Kappa:{}".format(TP, TN, FP, FN, Accuracy,
                                                                                                  Precision, Recall, Kappa))
    return TP, TN, FP, FN


if __name__ == '__main__':
    param1 = 1.0
    param2 = 1.0
    all_TP = []
    all_TN = []
    all_FP = []
    all_FN = []
    start_time = time.time()

    print("*=======================================================================*")
    path1 = './mydata/feature_matrices_5_fold/exclude_0_features.p.gz'
    path2 = './mydata/feature_matrices_5_fold/exclude_0miss_features.p.gz'
    path3 = './mydata/raw_data/drug_drug_synergy_original.csv'
    X_train, y_train, X_val, y_val, X_miss = get_data(path1, path2, path3, val_fold=0, norm='tanh')
    all_bal_train_test = get_balanced_sub_data2(X_train, y_train)
    temp_TP, temp_TN, temp_FP, temp_FN = bagging(all_bal_train_test, X_val, y_val, param1, param2, "0")
    print("spend_time:",time.time()-start_time)
    all_TP.append(temp_TP)
    all_TN.append(temp_TN)
    all_FP.append(temp_FP)
    all_FN.append(temp_FN)

    print("*=======================================================================*")
    path1 = './mydata/feature_matrices_5_fold/exclude_1_features.p.gz'
    path2 = './mydata/feature_matrices_5_fold/exclude_1miss_features.p.gz'
    path3 = './mydata/raw_data/drug_drug_synergy_original.csv'
    X_train, y_train, X_val, y_val, X_miss = get_data(path1, path2, path3, val_fold=1, norm='tanh')
    all_bal_train_test = get_balanced_sub_data2(X_train, y_train)
    temp_TP, temp_TN, temp_FP, temp_FN = bagging(all_bal_train_test, X_val, y_val, param1, param2, "1")
    print("spend time:", time.time()-start_time)
    all_TP.append(temp_TP)
    all_TN.append(temp_TN)
    all_FP.append(temp_FP)
    all_FN.append(temp_FN)

    print("*=======================================================================*")
    path1 = './mydata/feature_matrices_5_fold/exclude_2_features.p.gz'
    path2 = './mydata/feature_matrices_5_fold/exclude_2miss_features.p.gz'
    path3 = './mydata/raw_data/drug_drug_synergy_original.csv'
    X_train, y_train, X_val, y_val, X_miss = get_data(path1, path2, path3, val_fold=2, norm='tanh')
    all_bal_train_test = get_balanced_sub_data2(X_train, y_train)
    temp_TP, temp_TN, temp_FP, temp_FN = bagging(all_bal_train_test, X_val, y_val, param1, param2, "2")
    print("spend time:", time.time()-start_time)
    all_TP.append(temp_TP)
    all_TN.append(temp_TN)
    all_FP.append(temp_FP)
    all_FN.append(temp_FN)

    print("*=======================================================================*")
    path1 = './mydata/feature_matrices_5_fold/exclude_3_features.p.gz'
    path2 = './mydata/feature_matrices_5_fold/exclude_3miss_features.p.gz'
    path3 = './mydata/raw_data/drug_drug_synergy_original.csv'
    X_train, y_train, X_val, y_val, X_miss = get_data(path1, path2, path3, val_fold=3, norm='tanh')
    all_bal_train_test = get_balanced_sub_data2(X_train, y_train)
    temp_TP, temp_TN, temp_FP, temp_FN = bagging(all_bal_train_test, X_val, y_val, param1, param2, "3")
    print("spend time:", time.time()-start_time)
    all_TP.append(temp_TP)
    all_TN.append(temp_TN)
    all_FP.append(temp_FP)
    all_FN.append(temp_FN)

    print("*=======================================================================*")
    path1 = './mydata/feature_matrices_5_fold/exclude_4_features.p.gz'
    path2 = './mydata/feature_matrices_5_fold/exclude_4miss_features.p.gz'
    path3 = './mydata/raw_data/drug_drug_synergy_original.csv'
    X_train, y_train, X_val, y_val, X_miss = get_data(path1, path2, path3, val_fold=4, norm='tanh')
    all_bal_train_test = get_balanced_sub_data2(X_train, y_train)
    temp_TP, temp_TN, temp_FP, temp_FN = bagging(all_bal_train_test, X_val, y_val, param1, param2, "4")
    print("spend time:", time.time()-start_time)
    all_TP.append(temp_TP)
    all_TN.append(temp_TN)
    all_FP.append(temp_FP)
    all_FN.append(temp_FN)

    print("OverAll!")
    TP = sum(all_TP)
    TN = sum(all_TN)
    FP = sum(all_FP)
    FN = sum(all_FN)
    Accuracy = (TP + TN) / (TP+TN+FP+FN)
    Precision = TP / (TP + FP) if (TP + FP)!=0 else -1.234
    Recall = TP / (TP + FN) if (TP + FN)!=0 else -1.234
    p0 = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN)!=0 else -1.234
    pe = ((TP + FP) * (TP + FN) + (FP + TN) * (FN + TN)) / ((TP + TN + FP + FN) * (TP + TN + FP + FN))
    Kappa = (p0 - pe) / (1 - pe)
    print("TP:{}, TN:{}, FP:{}, FN:{}, Accuracy:{}, Precision:{}, Recall:{}, Kappa:{}".format(TP, TN, FP, FN, Accuracy,Precision, Recall, Kappa))
    print("*********************")
