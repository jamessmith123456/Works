# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 08:30:58 2018

@author: zhuangchi 
@email: 118106010716@njust.edu.cn
@phone: 15651963378/18851040596

"""

import xlrd
import jieba
import sys
import chardet
import codecs
import gensim
from gensim.models import Word2Vec
import numpy as np
import copy
import keras
from zipfile import ZipFile
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.layers import Dropout,Activation
from keras.utils import plot_model #这个是后来加上的，用于进行模型可视化的
from keras.callbacks import TensorBoard
import time
import matplotlib
from keras.layers import Dense, Activation  
import random
import math
import re
from sklearn import svm
from sklearn.cross_validation import train_test_split  
import numpy as np
import csv


###############################################################################
#下面是读取xlsx文件，录入相应的新闻和标题
def fileReader(filename):
    dataset = []  
    title = []
    content = []
    label = []
    ID = []
    workbook = xlrd.open_workbook(filename) #workbook不是一个具体的变量，是一个迭代器
    table = workbook.sheets()[0] #table也不是一个具体的变量，是一个迭代器
    if table.nrows != 0: #pytohn2中不等于还可以用<>表示
#        columns = len(table.row_values(0)) #xlsx文件的列数
        for row in range(table.nrows): #table.nrows表示行数
            dataset.append(table.row_values(row)) #table.row_values()表示值,是一个个方括号的list(补充一句，圆括号表示元组)
                                              #既然这里是方括号的list，那就好办了，直接list[0][1][2]不就分离出来了
            ID.append(table.row_values(row)[0])
            title.append(table.row_values(row)[1])
            content.append(table.row_values(row)[2])
            try:
                label.append(table.row_values(row)[3])
            except:
                print("测试数据没有标签得分！")
            finally:
                print("读取数据成功！")                                 
    else:
        print("该文件为空！")
    
    
    return dataset,title,content,label,ID
    

###############################################################################
def stopwords():
    f = open("F://华泰证券比赛//stopwords//stopwordsmyself.txt","rb")
    #这里记录踩过的坑,总是提示报错：'gbk' codec can't decode byte 0xff in position 0: illegal multibyte sequence
    #f.read()总是出错.解决办法：加上"b"采用读二进制的方式解决,"b"不能单独用,要和r w等一起用
    data = f.read()
    print(chardet.detect(data)) #输出结果显示编码格式为UTF-16
    try:
        f_text = data.decode('utf-16')
    finally:
        f.close()
    stoplist = f_text.split('\n')
    return stoplist

###############################################################################
#分词，将句子转化为单词词表
def sent2word(data,stoplist):
    """
    Segment a sentence to words
    Delete stopwords
    """
    newdata = []
    for i in range(len(data)):
        sentence = data[i]
        segList = jieba.cut(sentence)
        segResult = []
        for w in segList:
            segResult.append(w)

        newSent = []
        for word in segResult:
            if word in stoplist:
                # print "stopword: %s" % word
                continue
            else:
                newSent.append(word)
        newdata.append(newSent)
            
    return newdata



###############################################################################
#使用word2vec将词组转化为数值数据
def getWordVecs(wordList): #这个函数将词表中的所有单词都表示成数值
    vecs = []
    for word in wordList:
        word = word.replace('\n', '')
        try:
            vecs.append(model[word])
        except KeyError:
            continue
    # vecs = np.concatenate(vecs)
    return np.array(vecs, dtype = 'float')


def buildVecs(*data):
    Input = []
    Inputlabel = []
    if len(data) == 1:
        for i in range(len(data[0])):
            resultList = getWordVecs(data[0][i])
            if len(resultList) != 0: #这里得到的result实际是所有单词的向量n*m的形式，n代表单词数，m代表向量的维度
                resultArray = sum(np.array(resultList))/len(resultList)
                Input.append(resultArray)
    elif len(data) == 2:
        for i in range(len(data[0])):
            resultList = getWordVecs(data[0][i])
            if len(resultList) != 0: #这里得到的result实际是所有单词的向量n*m的形式，n代表单词数，m代表向量的维度
                resultArray = sum(np.array(resultList))/len(resultList)
                Input.append(resultArray)
                Inputlabel.append(data[1][i])
    
    return np.array(Input),np.array(Inputlabel)




def getWordVecs2(wordList): #这个函数将词表中的所有单词都表示成数值
    vecs = []
    for word in wordList:
        word = word.replace('\n', '')
        try:
            vecs.append(model2[word])
        except KeyError:
            continue
    # vecs = np.concatenate(vecs)
    return np.array(vecs, dtype = 'float')


def buildVecs2(*data):
    Input = []
    Inputlabel = []
    if len(data) == 1:
        for i in range(len(data[0])):
            resultList = getWordVecs2(data[0][i])
            if len(resultList) != 0: #这里得到的result实际是所有单词的向量n*m的形式，n代表单词数，m代表向量的维度
                resultArray = sum(np.array(resultList))/len(resultList)
                Input.append(resultArray)
    elif len(data) == 2:
        for i in range(len(data[0])):
            resultList = getWordVecs2(data[0][i])
            if len(resultList) != 0: #这里得到的result实际是所有单词的向量n*m的形式，n代表单词数，m代表向量的维度
                resultArray = sum(np.array(resultList))/len(resultList)
                Input.append(resultArray)
                Inputlabel.append(data[1][i])
    
    return np.array(Input),np.array(Inputlabel)








###############################################################################
#删除字符串操作
#s = '\tabc\t123\tisk'
#print(s.replace('\t', ''))
# 
#print("北门吹雪: http://www.cnblogs.com/2bjiujiu/")
# 
#import re
## 去除\r\n\t字符
#s = '\r\nabc\t123\nxyz'
#print(re.sub('[\r\n\t]', '', s))



#import string
#delEStr = string.punctuation + ' ' + string.digits  #ASCII 标点符号，空格和数字   
#delCStr = '《》（）&%￥#@！{}【】'   
def quchu(A):
    B = []
    for i in range(len(A)):
        temp = A[i]
        temp = temp.replace("<p class=Para>",'')
        temp = temp.replace("</p>",'')
        temp = temp.replace("\t\t",'')
        temp = temp.replace("<P class=Para>",'')
        temp = temp.replace("</P>",'')
        temp = temp.strip()
        temp = re.sub("[A-Za-z0-9\[\`\~\!\@\#\$\^\&\*\(\)\=\|\{\}\'\:\;\'\,\[\]\.\<\>\/\?\~\！\@\#\\\&\*\%]", "", temp)
        for j in range(10):
          temp = temp.replace(str(j),'')
        B.append(temp)
    
    return B




def Adaboost(x_train, y_train, x_eval, y_eval,x_test, y_test,N,TeK):
    #首先将训练集分为N份,分的时候注意各个类别都要按比例分,否则数据集更加不均衡了
    #那我随机用random分,感觉效果也应该不差？
    x_train_shape = x_train.shape
    x_train_row = x_train_shape[0]
    x_train_column = x_train_shape[1]
    
    y_train_shape = y_train.shape
    y_train_row = y_train_shape[0]
    
    y_train = y_train.reshape(y_train_row,1)
    temp = np.concatenate((x_train,y_train),axis=1)
    random.shuffle(temp)
    
    tempsize = temp.shape
    tempm = tempsize[0]
    tempn = tempsize[1]
    
    subm = math.floor(tempm/N)
    
    evalm = x_eval.shape[0]
    weight = np.ones((evalm,1))*(1/evalm)
    
    
    ResultPre = []
    Quanzhong = []
    ResultPreProb = []
    #下面开始Adaboost的正式迭代过程：
    for i in range(N):       
        #选定第i次的小子集,用以训练弱分类器
        subTrain = temp[i*subm:(i+1)*subm,0:TeK]
        subtrainlabel = temp[i*subm:(i+1)*subm,TeK]
    
        #训练弱分类器
        clf = svm.SVC(C=128, kernel='rbf', gamma=0.125, decision_function_shape='ovr',probability=True)
        clf.fit(subTrain, subtrainlabel.ravel())
        
        #将此弱分类器用于评价集
        y_eval_pre = clf.predict(x_eval)
        
        
        #开始更新权重
        weightemp = (y_eval_pre == y_eval)+0
        
        epsilom = 0
        for j in range(evalm):
            #计算该分类器权重
            epsilom = epsilom + weight[j]*weightemp[j]
            
        if epsilom < 0 or epsilom >0.5:
            weight = np.ones((evalm,1))*(1/evalm) #倘若该权重不在范围内,则重新恢复默认权重
#            epsilom = 0
        else:
            beta = epsilom/(1-epsilom) #如果权重符合要求,进行权重的更新
            su = 0
            for k in range(evalm):
                su = su + weight[k]*(beta**(1-weightemp[k]))
            
            for l in range(evalm):
                weight[l] = weight[l]*(beta**(1-weightemp[k]))/su
                
        y_test_pre = clf.predict(x_test) #计算该小分类器对于测试集的测试结果
        y_test_pre_prob = clf.predict_proba(x_test)
        
        ResultPre.append(y_test_pre)
        
        Quanzhong.append(epsilom)
        
        ResultPreProb.append(y_test_pre_prob)
        
    return ResultPre,Quanzhong,ResultPreProb
                
        
###############################################################################
#下面这个函数用于对Adaboost的结果进行综合,确定最终判定类别            
def getResult(ResultPre,Quanzhong,ResultPreProb):
#    ResultTemp = []
#    ResultTempProb = []
#    for i in range(len(Quanzhong)):
#        if i == 0:
#            temp = ResultPreProb[i] * Quanzhong[i]
#            he = temp
#        else:
#            he = he+temp
#    for i in range(len(he)):
#        ResultTempProb.append(he[i])
#        if he[i] >= 0:
#            ResultTemp.append(1)
#        else:
#            ResultTemp.append(-1)
    for i in range(len(Quanzhong)):
        temp = ResultPre[i] * Quanzhong[i]
        if i == 0:
            he = temp
        else:
            he = he + temp

#    return ResultTemp,ResultTempProb
    return he

###############################################################################
def MarticAndROC(y_test,ResultFinal):
    mylist = set(ResultFinal)
    classes = len(mylist)
    
    temp = np.zeros((classes,classes))
    
    for i in range(classes):
        for j in range(classes):
            for k in range(y_test.shape[0]):
                if y_test[k] == (i+1) and ResultFinal[k] == (j+1):
                    temp[i,j] = temp[i,j] + 1
    
    he = 0
    for i in range(classes):
        he = he + temp[i,i]
    
    Accuracy = he/y_test.shape[0]
    
    return temp,Accuracy

###############################################################################
#主函数入口
    
if __name__ == "__main__":
    Trdata,Trtitle,Trcontent,Trlabel,TrID = fileReader("F://华泰证券比赛//TRAIN.xlsx") #读取训练数据
    Tedata,Tetitle,Tecontent,Telabel,TeID = fileReader("F://华泰证券比赛//TEST.xlsx") #读取测试数据
    #下面这段想加入content内容进行处理.首先去除content里面的<P class=Para></P>
#    for i in range(len(Trcontent)):
#        Trcontent[i] = Trcontent[i].replace("<p class=Para>"," ")
#        Trcontent[i] = Trcontent[i].replace("</p>"," ")
#        Trcontent[i] = Trcontent[i].replace("\t\t"," ")
#        Trcontent[i] = Trcontent[i].strip()
#        
#    for i in range(len(Tecontent)):
#        Tecontent[i] = Tecontent[i].replace("<p class=Para>"," ")
#        Tecontent[i] = Tecontent[i].replace("</p>"," ")
#        Tecontent[i] = Tecontent[i].replace("\t\t"," ")
#        Tecontent[i] = Tecontent[i].strip()
        
    #合并新闻标题和新闻内容,也就是将两个字符串相连接
    for i in range(len(Trtitle)):
        Trtitle[i] = Trtitle[i] + Trcontent[i]
    
    for i in range(len(Tetitle)):
        Tetitle[i] = Tetitle[i] + Tecontent[i]   
     
    Trtitle = quchu(Trtitle)
    Tetitle = quchu(Tetitle)
    
    Trlabel.remove('score')
    StopList = stopwords() #构建停用词词表，以便分词

    #首先针对新闻标题title进行实验
    TrtitleWords = sent2word(Trtitle,StopList) #4416
    TetitleWords = sent2word(Tetitle,StopList) #1891
#    TrtitleWords.remove(['texttitlecontent'])
#    TetitleWords.remove(['texttitlecontent'])
    TrtitleWords.remove([])
    TetitleWords.remove([])
    
    AllWords = copy.deepcopy(TrtitleWords)
    AllWords.extend(TetitleWords) #6307
    
    #下面采用word2vec进行转换
    #model = Word2Vec(sentences, sg=1, size=100,  window=5,  min_count=5,  negative=3, sample=0.001, hs=1, workers=4)
    #model = gensim.models.KeyedVectors.load_word2vec_format("corpus.model.bin", binary = True)
    model = Word2Vec(AllWords, sg=1, size=120,  window=15,  min_count=5,  negative=3, sample=0.001, hs=1, workers=4)
    #window：表示当前词与预测词在一个句子中的最大距离是多少
    #min_count: 可以对字典做截断. 词频少于min_count次数的单词会被丢弃掉, 默认值为5
    #sg： 用于设置训练算法，默认为0，对应CBOW算法；sg=1则采用skip-gram算法。
    #size：是指特征向量的维度，默认为100。大的size需要更多的训练数据,但是效果会更好. 推荐值为几十到几百。
    #negative: 如果>0,则会采用negativesamp·ing，用于设置多少个noise words
    #sample: 高频词汇的随机降采样的配置阈值，默认为1e-3，范围是(0,1e-5)
    #hs: 如果为1则会采用hierarchica·softmax技巧。如果设置为0（defau·t），则negative sampling会被使用。
    #workers参数控制训练的并行数。
    
    Tr ,trlabel = buildVecs(TrtitleWords,Trlabel) #完美，数据得到了！
    Te ,telabel = buildVecs(TetitleWords) #完美，数据得到了！
    
    x_test = Te
    y_test = telabel
    x_train, x_eval, y_train, y_eval = train_test_split(Tr, trlabel, random_state=1, train_size=0.8)


    #应用上面的Adaboost算法,得到最终结果
    #想改进Adaboost算法，因为是三分类问题，想改进为3个二分类问题.
    classEs = list(set(y_train))
    
    RR = []
    RRProb = []
    for i in range(len(classEs)):
        baoliu = classEs[i]
        y_trainnew = np.array(y_train)
        y_testnew  = np.array(y_test)
        y_evalnew  = np.array(y_eval)
        for j in range(len(y_train)):
            if y_train[j] == baoliu:
                y_trainnew[j] = 1
            else:
                y_trainnew[j] = -1
                
        for j in range(len(y_eval)):
            if y_eval[j] == baoliu:
                y_evalnew[j] = 1
            else:
                y_evalnew[j] = -1       
        TeK = 120    
        N = 4
        ResultPre,Quanzhong,ResultPreProb = Adaboost(x_train, y_trainnew, x_eval, y_evalnew,x_test, y_testnew,N,TeK)
        he = getResult(ResultPre,Quanzhong,ResultPreProb)
        RR.append(he)
        
    kk =np.array(RR)
    jieguo = kk.argmax(axis=0)
    
    Tescore = ['SCORE']
    for i in range(jieguo.shape[0]):
        Tescore.append(jieguo[i])
        
    Result = []
    for i in range(len(Tescore)):
        temp = []
        temp.append(TeID[i])
        temp.append(Tescore[i])
        Result.append(temp)
    
    file_csv = codecs.open('result.csv','w+','utf-8')#追加
    writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    for data in Result:
        writer.writerow(data)
    print("保存文件成功，处理结束")