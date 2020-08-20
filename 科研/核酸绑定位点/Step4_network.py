#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import pandas as pd
plt.switch_backend('agg')
import os
#os.environ['CUDA_VISIBLE_DEVICES']='1'
import time
import math
import copy

def get_data(path):
    load_file = open(path, 'rb')
    all_feature1_train = pickle.load(load_file) #(7441,1024)
    all_feature2_train = pickle.load(load_file)  #(7441,1024)
    all_feature3_train = pickle.load(load_file)  #(7441,115,1279)
    all_feature1_test = pickle.load(load_file)  #(1862,1024)
    all_feature2_test = pickle.load(load_file)  #(1862,1024)
    all_feature3_test = pickle.load(load_file)  #(1862,115,1279)
    train_label = pickle.load(load_file)  #(7441,)
    test_label = pickle.load(load_file)  #(1862,)
    load_file.close()
    train_label_one_hot = np.zeros((train_label.shape[0],2))
    test_label_one_hot = np.zeros((test_label.shape[0], 2))
    for i in range(train_label.shape[0]):
        train_label_one_hot[i, int(train_label[i])] = 1
    for i in range(test_label.shape[0]):
        test_label_one_hot[i, int(test_label[i])] = 1
    return all_feature1_train, all_feature2_train, all_feature3_train, all_feature1_test, all_feature2_test, all_feature3_test, train_label_one_hot, test_label_one_hot

def get_balanced_sub_data2(all_feature1_train, all_feature2_train, all_feature3_train, train_label_one_hot):
    pos_index = []
    neg_index = []
    for i in range(all_feature1_train.shape[0]):
        if train_label_one_hot[i,0]==1:
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

        balanced_feature1_train = all_feature1_train[pos_index_copy, :]
        balanced_feature2_train = all_feature2_train[pos_index_copy, :]
        balanced_feature3_train = all_feature3_train[pos_index_copy, :, :]
        balanced_y_train = train_label_one_hot[pos_index_copy,:]
        pos_index_aft = []
        neg_index_aft = []
        for i in range(balanced_y_train.shape[0]):
            if balanced_y_train[i,0]==1:
                neg_index_aft.append(i)
            else:
                pos_index_aft.append(i)
        print("after subsampling, pos_num:{},neg_num:{}".format(len(pos_index_aft),len(neg_index_aft)))
        all_bal_train_test.append([balanced_feature1_train, balanced_feature2_train, balanced_feature3_train, balanced_y_train])
    return all_bal_train_test

def my_next_batch(all_feature1_train, all_feature2_train, all_feature3_train,train_label_one_hot, batch_size, batch_num):
    #这种batchsize采样方式，需要实现先把数据打乱
    start_index = (batch_num*batch_size)%all_feature1_train.shape[0]
    if (all_feature1_train.shape[0]-start_index)<=batch_size:
        end_index = all_feature1_train.shape[0]
    else:
        end_index = ((batch_num+1)*batch_size)%all_feature1_train.shape[0]
    batch_feature1_train = all_feature1_train[start_index:end_index,:]
    batch_feature2_train = all_feature2_train[start_index:end_index, :]
    batch_feature3_train = all_feature3_train[start_index:end_index, :, :]
    batch_target = train_label_one_hot[start_index:end_index,:]
    return batch_feature1_train, batch_feature2_train, batch_feature3_train, batch_target


def network(all_feature1_train, all_feature2_train, all_feature3_train, all_feature1_test, all_feature2_test, all_feature3_test, train_label_one_hot, test_label_one_hot, name):
    with tf.name_scope('input'):
        global_step = tf.Variable(0)
        input_x1 = tf.placeholder(tf.float32, [None, 1024])
        input_x2 = tf.placeholder(tf.float32, [None, 1024])
        input_x3 = tf.placeholder(tf.float32, [None, 115, 1279])
        output_y = tf.placeholder(tf.float32, [None, 2])
        input_x3_2 = tf.reshape(input_x3, [-1, 115, 1279, 1])

    with tf.name_scope('CNN'):
        conv1=tf.layers.conv2d(
            inputs=input_x3_2, #(None,1,3000,1)
            filters=8, #卷积核个数
            kernel_size=[3,3], #卷积核尺寸
            strides=1, #扫描步长
            padding='same', #边边补0 valid不需要补0，same需要补0，为了保证输入输出的尺寸一致,补多少不需要知道
            activation=tf.nn.relu #激活函数
        )
        conv2 = tf.layers.conv2d(
            inputs=conv1, #(None,115,1279,8)
            filters=8,
            kernel_size=[3, 3],
            strides=1,
            padding='same',
            activation=tf.nn.relu
        )
        pool1 = tf.layers.max_pooling2d(
            inputs=conv2, #(None,115,1279,8)
            pool_size=[2, 2], #过滤器的尺寸
            strides=2
        )
        conv3 = tf.layers.conv2d(
            inputs=pool1, #(None,58,639,8)
            filters=16,
            kernel_size=[3, 3],
            strides=1,
            padding='same',
            activation=tf.nn.relu
        )
        conv4 = tf.layers.conv2d(
            inputs=conv3, #(None,58,639,8)
            filters=16,
            kernel_size=[3, 3],
            strides=1,
            padding='same',
            activation=tf.nn.relu
        )
        pool2 = tf.layers.max_pooling2d(
            inputs=conv4, #(None,58,639,8)
            pool_size=[2, 2],
            strides=2
        )
        conv5 = tf.layers.conv2d(
            inputs=pool2, #(None,29,319,8)
            filters=32,
            kernel_size=[3, 3],
            strides=1,
            padding='same',
            activation=tf.nn.relu
        )
        conv6 = tf.layers.conv2d(
            inputs=conv5, #(None,29,319,8)
            filters=32,
            kernel_size=[3, 3],
            strides=1,
            padding='same',
            activation=tf.nn.relu
        )
        pool3 = tf.layers.max_pooling2d(
            inputs=conv6, #(None,29,319,16)
            pool_size=[2, 2],
            strides=2
        )
        conv7 = tf.layers.conv2d(
            inputs=pool3, #(None,29,319,8)
            filters=64,
            kernel_size=[3, 3],
            strides=1,
            padding='same',
            activation=tf.nn.relu
        )
        conv8 = tf.layers.conv2d(
            inputs=conv7, #(None,29,319,8)
            filters=64,
            kernel_size=[3, 3],
            strides=1,
            padding='same',
            activation=tf.nn.relu
        )
        pool4 = tf.layers.max_pooling2d(
            inputs=conv8, #(None,29,319,16)
            pool_size=[2, 2],
            strides=2
        )
        conv9 = tf.layers.conv2d(
            inputs=pool4, #(None,29,319,8)
            filters=64,
            kernel_size=[3, 3],
            strides=1,
            padding='same',
            activation=tf.nn.relu
        )
        conv10 = tf.layers.conv2d(
            inputs=conv9, #(None,29,319,8)
            filters=64,
            kernel_size=[3, 3],
            strides=1,
            padding='same',
            activation=tf.nn.relu
        )
        pool5 = tf.layers.max_pooling2d(
            inputs=conv10, #(None,29,319,16)
            pool_size=[2, 2],
            strides=2
        )
        flat = tf.reshape(pool5, [-1, 3 * 39 * 64]) #pool6.shape:(None,2,19,32)
        fully_flat = tf.concat([input_x1, input_x2, flat], axis=1) #(None,9536)
        dense = tf.layers.dense(
            inputs=fully_flat,
            units=2048,
            activation=tf.nn.relu
        )
        dense2 = tf.layers.dense(
            inputs=dense,
            units=512,
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
    loss = tf.reduce_mean(cost1)
    # 用Adam 优化器来最小化误差,学习率0.001 类似梯度下降
    initial_learning_rate = tf.Variable(0.001)

    boundaries = [10000, 13000]
    values = [0.001, 0.0001, 0.00001]
    learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
    # learning_rate = tf.train.exponential_decay(initial_learning_rate,
    #                                            global_step=global_step,
    #                                            decay_steps=50000, #总共训练2万次，每6千次减少为一半
    #                                            decay_rate=0.5)
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
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

        inner_start_time = time.time()
        for i in range(15000):
            batch_feature1_train, batch_feature2_train, batch_feature3_train, batch_target = my_next_batch(all_feature1_train, all_feature2_train, all_feature3_train,train_label_one_hot, 16, i)
            train_loss, test_accuracy, _, train_learning_rate = sess.run([loss, accuracy_op, train_op, learning_rate],
                                                                         feed_dict = {input_x1: batch_feature1_train ,
                                                                                      input_x2: batch_feature2_train ,
                                                                                      input_x3: batch_feature3_train ,
                                                                                      output_y:batch_target ,global_step:i})
            if i%1000==0 and i!=0:
                print("learning_rate:{},train 1000 steps, spends:{}".format(train_learning_rate, time.time()-inner_start_time))
                inner_start_time = time.time()
            if i%10000==0 and i!=0:
                print("iter:{}, learning_rate:{}".format(i, train_learning_rate))
                fig = plt.figure()  # 创建图片
                ax1 = fig.add_subplot(1, 2, 1)  # 显示损失值
                ax2 = fig.add_subplot(1, 2, 2)  # 显示测试集准确率
                # ------损失值图形--------
                ax1.plot(loss_array, "r--")
                ax1.set_title("loss value")
                # ------测试集准确率图形--------
                ax2.plot(test_ratio, "b--")
                ax2.set_title("test accuary")
                plt.savefig('my0628'+str(name)+'.png')
            loss_array.append(train_loss)
            test_ratio.append(test_accuracy)
        writer.close()

        test_output = np.zeros((test_label_one_hot.shape[0], 2))
        for i in range(test_label_one_hot.shape[0]):
            temp = sess.run(logits, {input_x1: all_feature1_test[i, :][np.newaxis, :],
                                     input_x2: all_feature2_test[i, :][np.newaxis, :],
                                     input_x3: all_feature3_test[i, :, :][np.newaxis, :, :]})
            test_output[i, :] = temp[0, :]
        inferenced_y = np.argmax(test_output, 1)

        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for i in range(inferenced_y.shape[0]):
            if inferenced_y[i] == 1 and test_label_one_hot[i, 1] == 1:
                TP += 1
            if inferenced_y[i] == 0 and test_label_one_hot[i, 0] == 1:
                TN += 1
            if inferenced_y[i] == 1 and test_label_one_hot[i, 0] == 1:
                FP += 1
            if inferenced_y[i] == 0 and test_label_one_hot[i, 1] == 1:
                FN += 1

        print('val_num:', test_label_one_hot.shape[0])
        print('right_num:', TP + TN)
        Accuracy = (TP + TN) / test_label_one_hot.shape[0]
        Precision = TP / (TP + FP) if (TP + FP) != 0 else -1.234
        Recall = TP / (TP + FN) if (TP + FN) != 0 else -1.234
        p0 = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else -1.234
        pe = ((TP + FP) * (TP + FN) + (FP + TN) * (FN + TN)) / ((TP + TN + FP + FN) * (TP + TN + FP + FN))
        Kappa = (p0 - pe) / (1 - pe)
        print("TP:{}, TN:{}, FP:{}, FN:{}, Accuracy:{}, Precision:{}, Recall:{}, Kappa:{}".format(TP, TN, FP, FN, Accuracy,
                                                                                                  Precision, Recall, Kappa))
        print("*********************")
        print("save picture!")
        fig = plt.figure()  # 创建图片
        ax1 = fig.add_subplot(1, 2, 1)  # 显示损失值
        ax2 = fig.add_subplot(1, 2, 2)  # 显示测试集准确率
        # ------损失值图形--------
        ax1.plot(loss_array, "r--")
        ax1.set_title("loss value")
        # ------测试集准确率图形--------
        ax2.plot(test_ratio, "b--")
        ax2.set_title("test accuary")
        plt.savefig('my0628'+str(name)+'.png')

        return Accuracy, inferenced_y, TP, TN, FP, FN

def bagging(all_bal_train_test, all_feature1_test, all_feature2_test, all_feature3_test, test_label_one_hot):
    all_ratio = []
    all_y = []
    for i in range(len(all_bal_train_test)):
        temp_all_feature1_train = all_bal_train_test[i][0]
        temp_all_feature2_train = all_bal_train_test[i][1]
        temp_all_feature3_train = all_bal_train_test[i][2]
        temp_train_label_one_hot = all_bal_train_test[i][3]
        tempacc, tempy, temp_TP, temp_TN, temp_FP, temp_FN = network(temp_all_feature1_train,
                                                                     temp_all_feature2_train,
                                                                     temp_all_feature3_train,
                                                                     all_feature1_test,
                                                                     all_feature2_test,
                                                                     all_feature3_test,
                                                                     temp_train_label_one_hot,
                                                                     test_label_one_hot,
                                                                     i)
        all_ratio.append(tempacc)
        all_y.append(tempy)
    save_file = open("bagging.pkl","wb")
    pickle.dump(all_ratio, save_file)
    pickle.dump(all_y, save_file)
    pickle.dump(test_label_one_hot, save_file)
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
    for i in range(test_label_one_hot.shape[0]):
        if all_result[i][0] > all_result[i][1] and test_label_one_hot[i, 0] == 1:
            TN += 1
        if all_result[i][0] > all_result[i][1] and test_label_one_hot[i, 1] == 1:
            FN += 1
        if all_result[i][0] < all_result[i][1] and test_label_one_hot[i, 0] == 1:
            FP += 1
        if all_result[i][0] < all_result[i][1] and test_label_one_hot[i, 1] == 1:
            TP += 1
    Accuracy = (TP + TN) / test_label_one_hot.shape[0]
    Precision = TP / (TP + FP) if (TP + FP) != 0 else -1.234
    Recall = TP / (TP + FN) if (TP + FN) != 0 else -1.234
    p0 = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else -1.234
    pe = ((TP + FP) * (TP + FN) + (FP + TN) * (FN + TN)) / ((TP + TN + FP + FN) * (TP + TN + FP + FN))
    Kappa = (p0 - pe) / (1 - pe)
    print("Finally_TP:{}, TN:{}, FP:{}, FN:{}, Accuracy:{}, Precision:{}, Recall:{}, Kappa:{}".format(TP, TN, FP, FN, Accuracy,
                                                                                                  Precision, Recall, Kappa))
    return TP, TN, FP, FN

if __name__ == '__main__':
    start_time = time.time()
    path = './train_and_test.pkl'
    all_feature1_train, all_feature2_train, all_feature3_train, all_feature1_test, all_feature2_test, all_feature3_test, train_label_one_hot, test_label_one_hot = get_data(path)
    print("read_data spends time:", time.time()-start_time)

    start_time = time.time()
    all_bal_train_test = get_balanced_sub_data2(all_feature1_train, all_feature2_train, all_feature3_train, train_label_one_hot)
    # for i in range(len(all_bal_train_test)):
    #     temp_all_bal_train_test = all_bal_train_test[i]
    #     save_file = open('temp_train'+str(i)+'.pkl','wb')
    #     pickle.dump(temp_all_bal_train_test, save_file)
    #     pickle.dump(all_feature1_test, save_file)
    #     pickle.dump(all_feature2_test, save_file)
    #     pickle.dump(all_feature3_test, save_file)
    #     pickle.dump(test_label_one_hot,save_file)
    #     save_file.close()
    temp_TP, temp_TN, temp_FP, temp_FN = bagging(all_bal_train_test, all_feature1_test, all_feature2_test, all_feature3_test, test_label_one_hot)
    #Accuracy, inferenced_y, TP, TN, FP, FN = network(all_feature1_train, all_feature2_train, all_feature3_train, all_feature1_test, all_feature2_test, all_feature3_test, train_label_one_hot, test_label_one_hot)
    print("network spends time:", time.time() - start_time)
    # save_file = open('result.pkl','wb')
    # pickle.dump(Accuracy, save_file)
    # pickle.dump(inferenced_y, save_file)
    # save_file.close()
    print("*=======================================================================*")