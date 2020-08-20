# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import pickle
import math

def get_my_data(path, window_size = 5, ratio=0.8):
    #path = 'Electricity price.txt'
    window_size = 5
    f = open(path,'r')
    content = list(f.readlines())
    all_data = []
    for item in content:
        if len(item.strip('\n').split(' '))==2 and ('' not in item.strip('\n').split(' ')):
            all_data.append(float(item.strip('\n').split(' ')[1]))
    all_X = []
    all_Y = []
    for i in range(window_size,len(all_data)):
        temp_example = []
        for j in range(1, window_size):
            temp_example.append(all_data[i-j])
        all_X.append(temp_example)
        all_Y.append(all_data[i])
    all_X = np.array(all_X)
    all_Y = np.array(all_Y)
    all_X = all_X[:,:,np.newaxis]
    all_Y = all_Y[:,np.newaxis]

    train_num = int(all_X.shape[0]*ratio)
    train_X = all_X[0:train_num,:,:] #(176,4,1)
    train_Y = all_Y[0:train_num,:] #(176,1)
    test_X = all_X[train_num+1:,:,:] #(43,4,1)
    test_Y = all_Y[train_num+1:,:] #(43,1)
    return train_X, train_Y, test_X, test_Y

# 随机取batch_size个训练样本 
def my_next_batch(train_data, train_target, batch_size):  
    #打乱数据集
    index = [ i for i in range(0,train_data.shape[0]) ]
    np.random.shuffle(index) 
    #建立batch_data与batch_target的空列表
    batch_data = []; 
    batch_target = [];  
    #向空列表加入训练集及标签
    for i in range(0,batch_size):  
        batch_data.append(train_data[index[i]]) 
        batch_target.append(train_target[index[i]])
    batch_data = np.array(batch_data) #float64(128,9,26)
    batch_target = np.array(batch_target) #int32 (128,2)
    return batch_data, batch_target

def cal_index(All_test_pre, test_Y):
    result = []
    y_pre = All_test_pre[0][1][0]
    temp = 0
    for i in range(len(All_test_pre[0][0])):
        temp += (y_pre[i,0]-test_Y[i,0])**2
        result.append(int(y_pre[i,0]))
    temp /= len(All_test_pre[0][0])
    temp = temp**0.5
    return result, temp

def single_layer_static_bi_lstm(input_x,n_steps,n_hidden):
    '''
    返回单层静态双向LSTM单元的输出，以及cell状态
    
    args:
        input_x:输入张量 形状为[batch_size,n_steps,n_input]
        n_steps:时序总数
        n_hidden：LSTM单元输出的节点个数 即隐藏层节点数
    '''
    #把输入input_x按列拆分，并返回一个有n_steps个张量组成的list 如batch_sizex28x28的输入拆成[(batch_size,28),((batch_size,28))....]
    #如果是调用的是静态rnn函数，需要这一步处理   即相当于把序列作为第一维度 
    input_x1 = tf.unstack(input_x,num=n_steps,axis=1)
    #正向
    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_hidden,forget_bias = 1.0)
    #反向
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_hidden,forget_bias = 1.0)
    #静态rnn函数传入的是一个张量list  每一个元素都是一个(batch_size,n_input)大小的张量  这里的输出hiddens是一个list 每一个元素都是前向输出,后向输出的合并
    hiddens,fw_state,bw_state = tf.contrib.rnn.static_bidirectional_rnn(cell_fw=lstm_fw_cell,cell_bw=lstm_bw_cell,inputs=input_x1,dtype=tf.float32)
    print('hiddens:\n',type(hiddens),len(hiddens),hiddens[0].shape,hiddens[1].shape)    #<class 'list'> 28 (?, 256) (?, 256)
    return hiddens,fw_state,bw_state

def single_layer_dynamic_bi_lstm(input_x,n_steps,n_hidden):
    '''
    返回单层动态双向LSTM单元的输出，以及cell状态
    
    args:
        input_x:输入张量 形状为[batch_size,n_steps,n_input]
        n_steps:时序总数
        n_hidden：gru单元输出的节点个数 即隐藏层节点数
    '''
    
     #正向
    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_hidden,forget_bias = 1.0)
    #反向
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_hidden,forget_bias = 1.0)

    
    #动态rnn函数传入的是一个三维张量，[batch_size,n_steps,n_input]  输出是一个元组 每一个元素也是这种形状
    hiddens,state = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,cell_bw=lstm_bw_cell,inputs=input_x,dtype=tf.float32)
    
    print('hiddens:\n',type(hiddens),len(hiddens),hiddens[0].shape,hiddens[1].shape)   #<class 'tuple'> 2 (?, 28, 128) (?, 28, 128)
    #按axis=2合并 (?,28,128) (?,28,128)按最后一维合并(?,28,256)
    hiddens = tf.concat(hiddens,axis=2)
    
    #注意这里输出需要转置  转换为时序优先的
    hiddens = tf.transpose(hiddens,[1,0,2])    
        
    return hiddens,state

def multi_layer_static_bi_lstm(input_x,n_steps,n_hidden):
    '''
    返回多层静态双向LSTM单元的输出，以及cell状态
    
    args:
        input_x:输入张量 形状为[batch_size,n_steps,n_input]
        n_steps:时序总数
        n_hidden：LSTM单元输出的节点个数 即隐藏层节点数
    '''
    
    #把输入input_x按列拆分，并返回一个有n_steps个张量组成的list 如batch_sizex28x28的输入拆成[(batch_size,28),((batch_size,28))....] 
    #如果是调用的是静态rnn函数，需要这一步处理   即相当于把序列作为第一维度 
    input_x1 = tf.unstack(input_x,num=n_steps,axis=1)

    stacked_fw_rnn = []
    stacked_bw_rnn = []
    for i in range(3):
        #正向
        stacked_fw_rnn.append(tf.contrib.rnn.BasicLSTMCell(num_units=n_hidden,forget_bias = 1.0))
        #反向
        stacked_bw_rnn.append(tf.contrib.rnn.BasicLSTMCell(num_units=n_hidden,forget_bias = 1.0))


    #静态rnn函数传入的是一个张量list  每一个元素都是一个(batch_size,n_input)大小的张量 这里的输出hiddens是一个list 每一个元素都是前向输出,后向输出的合并
    hiddens,fw_state,bw_state = tf.contrib.rnn.stack_bidirectional_rnn(stacked_fw_rnn,stacked_bw_rnn,inputs=input_x1,dtype=tf.float32)
        
    print('hiddens:\n',type(hiddens),len(hiddens),hiddens[0].shape,hiddens[1].shape)    #<class 'list'> 28 (?, 256) (?, 256)

    return hiddens,fw_state,bw_state

def multi_layer_dynamic_bi_lstm(input_x,n_steps,n_hidden):
    '''
    返回多层动态双向LSTM单元的输出，以及cell状态
    
    args:
        input_x:输入张量 形状为[batch_size,n_steps,n_input]
        n_steps:时序总数
        n_hidden：gru单元输出的节点个数 即隐藏层节点数
    '''    
    stacked_fw_rnn = []
    stacked_bw_rnn = []
    for i in range(3):
        #正向
        stacked_fw_rnn.append(tf.contrib.rnn.BasicLSTMCell(num_units=n_hidden,forget_bias = 1.0))
        #反向
        stacked_bw_rnn.append(tf.contrib.rnn.BasicLSTMCell(num_units=n_hidden,forget_bias = 1.0))
    tf.contrib.rnn.MultiRNNCell
    
    #动态rnn函数传入的是一个三维张量，[batch_size,n_steps,n_input]  输出也是这种形状，n_input变成了正向和反向合并之后的 即n_input*2
    hiddens,fw_state,bw_state = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(stacked_fw_rnn,stacked_bw_rnn,inputs=input_x,dtype=tf.float32)
    
    print('hiddens:\n',type(hiddens),hiddens.shape)   # <class 'tensorflow.python.framework.ops.Tensor'> (?, 28, 256)
        
    #注意这里输出需要转置  转换为时序优先的
    hiddens = tf.transpose(hiddens,[1,0,2])    
    
    return hiddens,fw_state,bw_state



def  lstm_electricty_prediction(flag, mode):
    '''
    arg:
        flags:表示构建的RNN结构是哪种
            1：单层静态双向LSTM
            2: 单层动态双向LSTM
            3：多层静态双向LSTM
            4: 多层动态双向LSTM
    '''
    '''
    1. 导入数据集
    '''
    tf.reset_default_graph()
    train_X, train_Y, test_X, test_Y = get_my_data('Electricity price.txt') #All_sequence_img:(34731,9,26) (34731,2)
    '''
定义参数，以及网络结构
    '''
    n_input = train_X.shape[2]             #LSTM单元输入节点的个数
    n_steps = train_X.shape[1]             #序列长度
    n_hidden = 128           #LSTM单元输出节点个数(即隐藏层个数)
    batch_size = 16         #小批量大小
    training_step = 2000     #迭代次数
    display_step  = 20      #显示步数
    learning_rate = 1e-3     #学习率
    n_classes = 1
    
    
    #定义占位符
    #batch_size：表示一次的批次样本数量batch_size  n_steps：表示时间序列总数  n_input：表示一个时序具体的数据长度  即一共28个时序，一个时序送入28个数据进入LSTM网络
    input_x = tf.placeholder(dtype=tf.float32,shape=[None,n_steps,n_input])
    input_y = tf.placeholder(dtype=tf.float32,shape=[None,1])
    
    
    #可以看做隐藏层
    if  flag == 1:
        print('单层静态双向LSTM网络：')
        hiddens,fw_state,bw_state = single_layer_static_bi_lstm(input_x,n_steps,n_hidden)
    elif flag == 2:
        print('单层动态双向LSTM网络：')
        hiddens,bw_state = single_layer_dynamic_bi_lstm(input_x,n_steps,n_hidden)
    elif flag == 3:
        print('多层静态双向LSTM网络：')
        hiddens,fw_state,bw_state = multi_layer_static_bi_lstm(input_x,n_steps,n_hidden)
    elif  flag == 4:
        print('多层动态双向LSTM网络：')
        hiddens,fw_state,bw_state = multi_layer_dynamic_bi_lstm(input_x,n_steps,n_hidden)


    #取LSTM最后一个时序的输出，然后经过全连接网络得到输出值
    output = tf.contrib.layers.fully_connected(inputs=hiddens[-1],num_outputs=n_classes)#,activation_fn = tf.nn.softmax
    #代价函数 J =-(Σy.logaL)/n    .表示逐元素乘
    cost = tf.reduce_mean(tf.square(input_y-output))
    #求解 根据mode来选择不同的梯度更新方式  BPTT:back propagation through time   BGD梯度下降 SGD随机梯度下降
    if mode==1:
        train = tf.train.AdamOptimizer(learning_rate).minimize(cost) #AdaGrad下降算法
    if mode==2:
        train = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(cost)  # 动量法下降算法
    if mode==3:
        train = tf.train.AdadeltaOptimizer(learning_rate).minimize(cost) #Adadelta优化算法
    if mode==4:
        train = tf.train.RMSPropOptimizer(learning_rate).minimize(cost) #RMS优化器

    #创建list 保存每一迭代的结果
    test_accuracy_list = []
    All_test_pre = []
    test_cost_list=[]
    #for index in range(5):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())   #初始化变量
        for i in range(training_step):
            #x_batch,y_batch = mnist.train.next_batch(batch_size = batch_size) #(128*784) (128*10)y_batch实际就是0-1编码
            x_batch,y_batch = my_next_batch(train_X, train_Y, batch_size)
            #开始训练
            train.run(feed_dict={input_x:x_batch,input_y:y_batch})
    #全部训练完成做测试  分成200次，一次测试50个样本
    #输出测试机准确率   如果一次性全部做测试，内容不够用会出现OOM错误。所以测试时选取比较小的mini_batch来测试
        #x_batch,y_batch = mnist.test.next_batch(batch_size = 50)
        x_batch,y_batch = test_X,test_Y
        #Reshape data to get 28 seq of 28 elements
        #x_batch = x_batch.reshape([-1,n_steps,n_input])
        test_cost = sess.run([cost],feed_dict={input_x:x_batch,input_y:y_batch})
        test_output = sess.run([output],feed_dict={input_x:x_batch,input_y:y_batch})
        All_test_pre.append([y_batch,test_output])
        test_cost_list.append(test_cost)
    result , mse = cal_index(All_test_pre, test_Y)
    #print("MSE:", mse)
    return result, test_cost_list, mse

if __name__ == '__main__':
    optimizer_mode = {1:'AdaGrad优化算法',2:'Momentum优化算法',3:'Adadelta优化算法',4:'RMSProp优化算法'}
    for item in optimizer_mode.keys():
        result, test_cost_list, mse = lstm_electricty_prediction(1, item) #单层静态双向LSTM网络
        print(optimizer_mode[item],"mse:",mse)
