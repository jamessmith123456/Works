#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
#import pandas as pd
plt.switch_backend('agg')
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import time
import math
import copy
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# import warnings
# warnings.filterwarnings('ignore') # 注：放的位置也会影响效果，真是奇妙的代码
def binary(data, threshold = 30):
    data[ data < threshold ] = 0
    data[ data >= threshold ] = 1
    pos_num = 0
    neg_num = 0
    for i in range(data.shape[0]):
        if data[i] == 1:
            pos_num += 1
        if data[i] == 0:
            neg_num += 1
    print("pos_example_num:{},neg_example_num:{}".format(pos_num, neg_num))
    return data


def get_data(path):
    load_file = open("./cellline_pssm_pss_feature30.pkl", 'rb')
    all_cellline = pickle.load(load_file)  #21种疾病
    all_cellline_feature = pickle.load(load_file) #每个疾病是键  键值是(23,1279)/(115,1279)/(115,1279)的array
    load_file.close()

    #load_file = open("morgan_feature.pkl","rb")
    load_file = open("compound_feature.pkl", "rb")
    all_drug = pickle.load(load_file) #35种药物  每种药物的键值是它的SMILE分子格式
    all_feature_drug = pickle.load(load_file) #33种药物的名字
    all_drug_Estate = pickle.load(load_file) #79   33种药物
    all_drug_Daylight = pickle.load(load_file) #2048 33种药物
    all_drug_MACC = pickle.load(load_file) #167   33种药物
    all_drug_Morgan = pickle.load(load_file) #1024 33种药物
    load_file.close()

    #with open('F:/DrugScreening2/mydata2/raw_data/drug_drug_synergy_original.csv','r') as f:
    with open('./drug_drug_synergy_original.csv', 'r') as f:
        content = f.readlines()
    content = content[1:]
    all_sample = []
    all_label = []
    all_feature1_estate = []
    all_feature2_estate = []
    all_feature1_Daylight = []
    all_feature2_Daylight = []
    all_feature1_MACC = []
    all_feature2_MACC = []
    all_feature1_Morgan = []
    all_feature2_Morgan = []
    all_feature3 = []
    count = 0
    for i in range(len(content)):
        temp = content[i].split(',')
        if temp[1] in all_drug_Morgan.keys() and temp[2] in all_drug_Morgan.keys() and temp[3] in all_cellline:
            count += 1 #一共9303个
            temp_sample = [temp[1], temp[2], temp[3], float(temp[4])]
            all_sample.append(temp_sample) #一共9303个元素 每个元素是一个为4的list  ['5-FU', 'ABT-888', 'A2058', 7.6935]
            all_label.append(float(temp[4]))
            #print("drugA: {}, drugB:{}".format(temp[1], temp[2]))
            all_feature1_estate.append(all_drug_Estate[temp[1]]) #9303个元素 每个元素为79
            all_feature2_estate.append(all_drug_Estate[temp[2]]) ##9303个元素 每个元素为79
            all_feature1_Daylight.append(all_drug_Daylight[temp[1]]) #9303个元素 每个元素为2048
            all_feature2_Daylight.append(all_drug_Daylight[temp[2]]) #9303个元素 每个元素为2048
            all_feature1_MACC.append(all_drug_MACC[temp[1]])
            all_feature2_MACC.append(all_drug_MACC[temp[2]])
            all_feature1_Morgan.append(all_drug_Morgan[temp[1]])
            all_feature2_Morgan.append(all_drug_Morgan[temp[2]])
            all_feature3.append(all_cellline_feature[temp[3]]) #9303个元素 每个元素又包含230个长为1279的list

    all_feature1_estate = np.array(all_feature1_estate, dtype=np.float16) #(9303,79)
    all_feature2_estate = np.array(all_feature2_estate, dtype=np.float16) #(9303,79)
    all_feature1_Daylight = np.array(all_feature1_Daylight, dtype=np.float16) #(9303,2048)
    all_feature2_Daylight = np.array(all_feature2_Daylight, dtype=np.float16) #(9303,2048)
    all_feature1_MACC = np.array(all_feature1_MACC, dtype=np.float16) #(9303,167)
    all_feature2_MACC = np.array(all_feature2_MACC, dtype=np.float16) #(9303,167)
    all_feature1_Morgan = np.array(all_feature1_Morgan, dtype=np.float16) #(9303,1024)
    all_feature2_Morgan = np.array(all_feature2_Morgan, dtype=np.float16) #(9303,1024)
    all_feature3 = np.array(all_feature3, dtype=np.float16) #(9303,230,1279)

    all_label = np.array(all_label, dtype=np.float16)
    all_label = binary(all_label, 30) #正样本880 负样本8423


    load_file = open(path, 'rb')
    temp_train_index = pickle.load(load_file)
    temp_test_index = pickle.load(load_file)
    load_file.close()
    all_feature1_train_estate = all_feature1_estate[temp_train_index, :]
    all_feature2_train_estate = all_feature2_estate[temp_train_index, :]

    all_feature1_train_Daylight = all_feature1_Daylight[temp_train_index, :]
    all_feature2_train_Daylight = all_feature2_Daylight[temp_train_index, :]

    all_feature1_train_MACC = all_feature1_MACC[temp_train_index, :]
    all_feature2_train_MACC = all_feature2_MACC[temp_train_index, :]

    all_feature1_train_Morgan = all_feature1_Morgan[temp_train_index, :]
    all_feature2_train_Morgan = all_feature2_Morgan[temp_train_index, :]

    all_feature3_train = all_feature3[temp_train_index, :, :]

    all_feature1_test_estate = all_feature1_estate[temp_test_index, :]
    all_feature2_test_estate = all_feature2_estate[temp_test_index, :]

    all_feature1_test_Daylight = all_feature1_Daylight[temp_test_index, :]
    all_feature2_test_Daylight = all_feature2_Daylight[temp_test_index, :]

    all_feature1_test_MACC = all_feature1_MACC[temp_test_index, :]
    all_feature2_test_MACC = all_feature2_MACC[temp_test_index, :]

    all_feature1_test_Morgan = all_feature1_Morgan[temp_test_index, :]
    all_feature2_test_Morgan = all_feature2_Morgan[temp_test_index, :]

    all_feature3_test = all_feature3[temp_test_index, :, :]

    train_label = all_label[temp_train_index]
    test_label = all_label[temp_test_index]
    print(path)
    print("all_feature1_train_estate.shape:", all_feature1_train_estate.shape)
    print("all_feature2_train_estate.shape:", all_feature2_train_estate.shape)
    print("all_feature1_train_Daylight.shape:", all_feature1_train_Daylight.shape)
    print("all_feature2_train_Daylight.shape:", all_feature2_train_Daylight.shape)
    print("all_feature1_train_MACC.shape:", all_feature1_train_MACC.shape)
    print("all_feature2_train_MACC.shape:", all_feature2_train_MACC.shape)
    print("all_feature1_train_Morgan.shape:", all_feature1_train_Morgan.shape)
    print("all_feature2_train_Morgan.shape:", all_feature2_train_Morgan.shape)

    print("all_feature3_train.shape:", all_feature3_train.shape)

    print("all_feature1_test_estate.shape:", all_feature1_test_estate.shape)
    print("all_feature2_test_estate.shape:", all_feature2_test_estate.shape)
    print("all_feature1_test_Daylight.shape:", all_feature1_test_Daylight.shape)
    print("all_feature2_test_Daylight.shape:", all_feature2_test_Daylight.shape)
    print("all_feature1_test_MACC.shape:", all_feature1_test_MACC.shape)
    print("all_feature2_test_MACC.shape:", all_feature2_test_MACC.shape)
    print("all_feature1_test_Morgan.shape:", all_feature1_test_Morgan.shape)
    print("all_feature2_test_Morgan.shape:", all_feature2_test_Morgan.shape)

    print("all_feature3_test.shape:", all_feature3_test.shape)
    print("train_label.shape:", train_label.shape)
    print("test_label.shape:", test_label.shape)

    train_label_one_hot = np.zeros((train_label.shape[0],2))
    test_label_one_hot = np.zeros((test_label.shape[0], 2))
    for i in range(train_label.shape[0]):
        train_label_one_hot[i, int(train_label[i])] = 1
    for i in range(test_label.shape[0]):
        test_label_one_hot[i, int(test_label[i])] = 1

    all_feature1_train = all_feature1_train_MACC
    all_feature2_train = all_feature2_train_MACC
    all_feature1_test = all_feature1_test_MACC
    all_feature2_test = all_feature2_test_MACC
    print("*****************************************************")
    print("all_feature1_train.shape:", all_feature1_train.shape)
    print("all_feature2_train.shape:", all_feature1_train.shape)
    print("all_feature3_train.shape:", all_feature3_train.shape)
    print("all_feature1_test.shape:", all_feature1_test.shape)
    print("all_feature2_test.shape:", all_feature2_test.shape)
    print("all_feature3_test.shape:", all_feature3_test.shape)
    print("train_label_one_hot.shape:", train_label_one_hot.shape)
    print("test_label_one_hot.shape:", test_label_one_hot.shape)
    print("*****************************************************")
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

def network(all_feature1_train, all_feature2_train, all_feature3_train, all_feature1_test, all_feature2_test, all_feature3_test, train_label_one_hot, test_label_one_hot, name, fold):
    tf.reset_default_graph()
    global_step = tf.Variable(0)
    input_x1 = tf.placeholder(tf.float32, [None, 167], name="input_x1")
    input_x2 = tf.placeholder(tf.float32, [None, 167], name="input_x2")
    input_x3 = tf.placeholder(tf.float32, [None, 690, 1279], name="input_x3")
    output_y = tf.placeholder(tf.float32, [None, 2], name="input_y")
    input_x3_2 = tf.reshape(input_x3, [-1, 690, 1279, 1], name="input_x4")
    #gedcnnpre = tf.placeholder(tf.float32, [None, 2], name="gedcnnpre1")

    #with tf.name_scope('CNN'):
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
        filters=128,
        kernel_size=[3, 3],
        strides=1,
        padding='same',
        activation=tf.nn.relu
    )
    conv10 = tf.layers.conv2d(
        inputs=conv9, #(None,29,319,8)
        filters=128,
        kernel_size=[3, 3],
        strides=1,
        padding='same',
        activation=tf.nn.relu
    )
    pool5 = tf.layers.max_pooling2d(
        inputs=conv10, #(None,29,319,16)
        pool_size=[1, 2],
        strides=2
    )
    # conv11 = tf.layers.conv2d(
    #     inputs=pool5, #(None,29,319,8)
    #     filters=512,
    #     kernel_size=[3, 3],
    #     strides=1,
    #     padding='same',
    #     activation=tf.nn.relu
    # )
    # conv12 = tf.layers.conv2d(
    #     inputs=conv11, #(None,29,319,8)
    #     filters=512,
    #     kernel_size=[3, 3],
    #     strides=1,
    #     padding='same',
    #     activation=tf.nn.relu
    # )
    # pool6 = tf.layers.max_pooling2d(
    #     inputs=conv12, #(None,29,319,16)
    #     pool_size=[2, 2],
    #     strides=2
    # )
    flat = tf.reshape(pool5, [-1, 22 * 39 * 128]) #pool6.shape:(None,2,19,32)
    fully_flat = tf.concat([input_x1, input_x2, flat], axis=1) #(None,9536)
    dense = tf.layers.dense(
        inputs=fully_flat,
        units=4096,
        activation=tf.nn.relu
    )
    dense1 = tf.layers.dense(
        inputs=dense,
        units=1024,
        activation=tf.nn.relu
    )
    dense2 = tf.layers.dense(
        inputs=dense1,
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
    gedcnnpre = tf.layers.dense(
        inputs=dense3,
        units=2,
        name="gedcnnprefuck"
    )
    # loss = tf.losses.softmax_cross_entropy(onehot_labels=output_y,
    #                                        logits=logits)
    cost1 = tf.nn.sigmoid_cross_entropy_with_logits(logits=gedcnnpre, labels=output_y)
    loss = tf.reduce_mean(cost1)
    # 用Adam 优化器来最小化误差,学习率0.001 类似梯度下降
    initial_learning_rate = tf.Variable(0.001)

    boundaries = [60000, 110000]
    values = [0.01, 0.001, 0.0001]
    learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
    # learning_rate = tf.train.exponential_decay(initial_learning_rate,
    #                                            global_step=global_step,
    #                                            decay_steps=50000, #总共训练2万次，每6千次减少为一半
    #                                            decay_rate=0.5)
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    accuracy_op=tf.metrics.accuracy(
        labels=tf.argmax(output_y,axis=1),
        predictions=tf.argmax(gedcnnpre,axis=1)
    )[1]

    init = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.per_process_gpu_memory_fraction = 1.0  # 分配50%
    with tf.Session(config=tf_config) as sess:
        print(gedcnnpre.name)
        loss_array = []
        test_ratio = []
        sess.run(init)
        writer = tf.summary.FileWriter('logs/', sess.graph) #tf.get_default_graph()

        inner_start_time = time.time()
        Final_all_index = []
        for i in range(100000):
            batch_feature1_train, batch_feature2_train, batch_feature3_train, batch_target = my_next_batch(all_feature1_train, all_feature2_train, all_feature3_train,train_label_one_hot, 32, i)
            train_loss, test_accuracy, _, train_learning_rate = sess.run([loss, accuracy_op, train_op, learning_rate],
                                                                         feed_dict = {input_x1: batch_feature1_train ,
                                                                                      input_x2: batch_feature2_train ,
                                                                                      input_x3: batch_feature3_train ,
                                                                                      output_y:batch_target ,global_step:i})
            if i%5000==0 and i!=0:
                tf.train.Saver().save(sess, "./model/GEDCNN")
                pass
                #print("learning_rate:{},train 1000 steps, spends:{}".format(train_learning_rate, time.time()-inner_start_time))
                #inner_start_time = time.time()
            if i%10000==0 and i!=0:
                print("===============================")
                #怎么下面这种写法  直接就把学习率显示为0了...
                print("iter:{}, learning_rate:{}, spends:{}".format(i, round(train_learning_rate,6), round(time.time()-inner_start_time,3)))
                fig = plt.figure()  # 创建图片
                ax1 = fig.add_subplot(1, 2, 1)  # 显示损失值
                ax2 = fig.add_subplot(1, 2, 2)  # 显示测试集准确率
                # ------损失值图形--------
                ax1.plot(loss_array, "r--")
                ax1.set_title("loss value")
                # ------测试集准确率图形--------
                ax2.plot(test_ratio, "b--")
                ax2.set_title("test accuary")
                plt.savefig('Example0705'+str(fold)+'_'+str(name)+'.png')

                test_output = np.zeros((test_label_one_hot.shape[0], 2))
                for i in range(test_label_one_hot.shape[0]):
                    temp = sess.run(gedcnnpre, {input_x1: all_feature1_test[i, :][np.newaxis, :],
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

                #print('val_num:', test_label_one_hot.shape[0])
                #print('right_num:', TP + TN)
                Accuracy = (TP + TN) / (TP+TN+FP+FN)
                Precision = TP / (TP + FP) if (TP + FP) != 0 else -1.234
                Recall = TP / (TP + FN) if (TP + FN) != 0 else -1.234
                p0 = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else -1.234
                pe = ((TP + FP) * (TP + FN) + (FP + TN) * (FN + TN)) / ((TP + TN + FP + FN) * (TP + TN + FP + FN))
                Kappa = (p0 - pe) / (1 - pe)
                Accuracy = round(Accuracy, 3)
                Precision = round(Precision, 3)
                Recall = round(Recall, 3)
                Kappa = round(Kappa, 3)
                MCC = (TP * TN - FP * FN) / (((TP + FP) * (TN + FN) * (TP + FN) * (TN + FP)) ** 0.5)
                MCC = round(MCC, 3)
                print("TP:{}, TN:{}, FP:{}, FN:{}, Accuracy:{}, Precision:{}, Recall:{}, Kappa:{}, MCC:{}".format(TP, TN, FP, FN,
                                                                                                          Accuracy,
                                                                                                          Precision, Recall,
                                                                                                          Kappa, MCC))
                Final_all_index.append([TP, TN, FP, FN, Accuracy, Precision, Recall, Kappa, MCC, inferenced_y, test_output, loss_array, test_ratio, test_label_one_hot])
            loss_array.append(train_loss)
            test_ratio.append(test_accuracy)
        writer.close()

        test_output = np.zeros((test_label_one_hot.shape[0], 2))
        for i in range(test_label_one_hot.shape[0]):
            temp = sess.run(gedcnnpre, {input_x1: all_feature1_test[i, :][np.newaxis, :],
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

        #print('val_num:', test_label_one_hot.shape[0])
        #print('right_num:', TP + TN)
        Accuracy = (TP + TN) / test_label_one_hot.shape[0]
        Precision = TP / (TP + FP) if (TP + FP) != 0 else -1.234
        Recall = TP / (TP + FN) if (TP + FN) != 0 else -1.234
        p0 = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else -1.234
        pe = ((TP + FP) * (TP + FN) + (FP + TN) * (FN + TN)) / ((TP + TN + FP + FN) * (TP + TN + FP + FN))
        Kappa = (p0 - pe) / (1 - pe)
        Accuracy = round(Accuracy, 3)
        Precision = round(Precision, 3)
        Recall = round(Recall, 3)
        Kappa = round(Kappa, 3)
        MCC = (TP * TN - FP * FN) / (((TP + FP) * (TN + FN) * (TP + FN) * (TN + FP)) ** 0.5)
        MCC = round(MCC, 3)
        print("TP:{}, TN:{}, FP:{}, FN:{}, Accuracy:{}, Precision:{}, Recall:{}, Kappa:{}, MCC:{}".format(TP, TN, FP, FN,
                                                                                                        Accuracy,
                                                                                                        Precision,
                                                                                                        Recall,
                                                                                                        Kappa, MCC))
        Final_all_index.append([TP, TN, FP, FN, Accuracy, Precision, Recall, Kappa, MCC, inferenced_y, test_output, loss_array, test_ratio, test_label_one_hot])
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
        plt.savefig('Example0705'+str(fold)+'_'+str(name)+'.png')

        #return Accuracy, inferenced_y, TP, TN, FP, FN, test_output
        return Final_all_index

if __name__ == '__main__':
    for iter in range(5):
        print("iter {} begin".format(iter))
        start_time = time.time()
        path = './Estate_train_and_test'+str(iter)+'.pkl'
        all_feature1_train, all_feature2_train, all_feature3_train, all_feature1_test, all_feature2_test, all_feature3_test, train_label_one_hot, test_label_one_hot = get_data(path)
        print("all_feature1_train.shape:", all_feature1_train.shape)
        print("all_feature2_train.shape:", all_feature2_train.shape)
        print("all_feature3_train.shape:", all_feature3_train.shape)
        print("all_feature1_test.shape:", all_feature1_test.shape)
        print("all_feature2_test.shape:", all_feature2_test.shape)
        print("all_feature3_test.shape:", all_feature3_test.shape)
        print("train_label_one_hot.shape:", train_label_one_hot.shape)
        print("test_label_one_hot.shape:", test_label_one_hot.shape)

        print("read_data spends time:", time.time()-start_time)

        start_time = time.time()
        Final_all_index = network(all_feature1_train,all_feature2_train,all_feature3_train,
                                  all_feature1_test,all_feature2_test,all_feature3_test,
                                   train_label_one_hot,test_label_one_hot,iter, 'wrong')
        save_file = open("Compound2_temp_fold"+str(iter)+".pkl",'wb')
        pickle.dump(Final_all_index, save_file)
        save_file.close()

        print("network spends time:", time.time() - start_time)
        print("****************************************************************************************************************")
