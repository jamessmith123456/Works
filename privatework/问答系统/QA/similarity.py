import jieba
from gensim import corpora,models,similarities
from collections import defaultdict   #用于创建一个空的字典，在后续统计词频可清理频率少的词语
import csv
import numpy as np
from sklearn.naive_bayes import MultinomialNB

def get_alld(path):
    #path = "question_best.csv"
    f = csv.reader(open(path, 'r', encoding="utf-8"))
    all_d = []
    for i in f:
        all_d.append(list(i)[0]+'/'+list(i)[1])
    all_d = all_d[1:]
    return all_d

def get_similraity_index(all_d , target_d):
#def get_similraity_index(path, target_d):
    documents = []
    for d in all_d:
        data = jieba.cut(d)
        data1 = ""
        for i in data:
            data1 += i + " "
        documents.append(data1)

    texts = [[word for word in document.split()] for document in documents] #764个词汇
    #计算词语的频率
    frequency=defaultdict(int)
    for text in texts:
        for word in text:
            frequency[word]+=1
    '''
    #对频率低的词语进行过滤（可选）
    texts=[[word for word in text if frequency[word]>10] for text in texts]
    '''
    #通过语料库将文档的词语进行建立词典
    dictionary=corpora.Dictionary(texts)
    dictionary.save("./dict.txt")#可以将生成的词典进行保存

    #计算要对比的文本
    target_data=jieba.cut(target_d)
    target_data1=""
    for i in target_data:
        target_data1+=i+" "
    #将要对比的文档通过doc2bow转化为稀疏向量
    new_xs=dictionary.doc2bow(target_data1.split())
    #对语料库进一步处理，得到新语料库
    corpus=[dictionary.doc2bow(text)for text in texts]
    #将新语料库通过tf-idf model 进行处理，得到tfidf
    tfidf=models.TfidfModel(corpus)
    #通过token2id得到特征数
    featurenum=len(dictionary.token2id.keys())
    #稀疏矩阵相似度，从而建立索引
    index=similarities.SparseMatrixSimilarity(tfidf[corpus],num_features=featurenum)
    #得到最终相似结果
    sim=index[tfidf[new_xs]]
    #print(sim)
    target_index = np.argmax(sim)

    all_cos = []
    for i in range(len(all_d)):
        temp2 = [item[1] for item in tfidf[new_xs]]
        known_d = all_d[i]
        known_data=jieba.cut(known_d)
        known_data1=""
        for ii in known_data:
            known_data1+=ii+" "
        #将要对比的文档通过doc2bow转化为稀疏向量
        known_new_xs=dictionary.doc2bow(known_data1.split())
        temp1 = [item[1] for item in tfidf[known_new_xs]]
        temp2 = np.array(temp2)
        temp1 = np.array(temp1)
        if temp2.shape[0]>temp1.shape[0]:
            temp = np.zeros(temp2.shape)
            for k in range(temp1.shape[0]):
                temp[k] = temp1[k]
            all_cos.append(abs(np.dot(temp, temp2) / (np.linalg.norm(temp) * np.linalg.norm(temp2))))
        else:
            temp = np.zeros(temp1.shape)
            for k in range(temp2.shape[0]):
                temp[k] = temp2[k]
            all_cos.append(abs(np.dot(temp, temp1) / (np.linalg.norm(temp) * np.linalg.norm(temp1))))
    cos_target_index = all_cos.index(max(all_cos))

    # clf = MultinomialNB(alpha=0.01) #贝叶斯也不好,还是tf-idf好
    # clf.fit(train_data, y_train)
    return all_d[target_index].split("/")[0], all_d[cos_target_index].split("/")[0]