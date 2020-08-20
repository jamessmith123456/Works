import jieba
import gensim
from gensim import corpora
from gensim import similarities
import pandas as pd
from utils import *

def get_data(intro_list):
    # 中文预处理
    new_list = []
    for intro in intro_list:
        new_list.append(format_str(intro))

    # 进行分词
    fenci_list = []
    fenci_list = fenci(new_list)

    # 去除停用词
    with open('./stopwords.txt', 'r', encoding='utf-8') as f:
        stopwords_list = []
        for line in f.readlines():
            stopwords_list.append(line.strip())

    clean_list = drop_stopwords(fenci_list, stopwords_list)
    return clean_list


# 中文文本预处理
def is_cnen(uchar):
    # 只保留汉字与英文,去掉了数字标点和特殊字符
    if (uchar >= u'\u4e00' and uchar <= u'\u9fa5') or (uchar >= u'\u0041' and uchar <= u'\u005A') or (
            uchar >= u'\u0061' and uchar <= u'\u007A'):
        return True
    else:
        return False


def format_str(content):
    content_str = ''
    for i in content:
        if is_cnen(i):
            content_str = content_str + ｉ
    return content_str


# 分词
def fenci(new_list):
    cut_words = map(lambda s: list(jieba.cut(s)), new_list)
    return list(cut_words)


# 去除停用词
def drop_stopwords(fenci_list, stopwords):
    fenci_list_clean = []
    for line in fenci_list:
        line_clean = []
        for word in line:
            if word in stopwords:
                continue
            line_clean.append(word)
        fenci_list_clean.append(line_clean)
    return fenci_list_clean


def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim

# 读取数据，进行中文预处理，得到词语字典，将文档变成DT矩阵
# 读取frame中的app概要说明
game_frame = pd.read_csv("../data/game.csv")
intro_list = game_frame['概要说明'].values.tolist()

# 得到的clean_list是预处理完成的词语列表
# get_data函数对读入数据进行预处理，保留汉字与英文字母，分词，去除停用词
clean_list = get_data(intro_list)

# 下一步准备 Document-Term 矩阵
# 创建语料的词语词典，每个单独的词语都会被赋予一个索引
dictionary = corpora.Dictionary(clean_list)

# 使用上面的词典，将转换文档列表（语料）变成 DT 矩阵
doc_term_matrix = [dictionary.doc2bow(doc) for doc in clean_list]

# 创建tfidf对象
# LSI LDA 等模型通常用bow向量或是tfidf向量作为语料输入，上面的doc_term_matrix就是bow向量
tfidf = gensim.models.TfidfModel(doc_term_matrix)
corpus_tfidf = tfidf[doc_term_matrix]

# 构建LDA模型进行训练
Lda = gensim.models.ldamodel.LdaModel
ldamodel = Lda(corpus_tfidf, num_topics=3, id2word=dictionary, passes=50)

# 构建LSI模型进行训练
Lsi = gensim.models.LsiModel
lsimodel = Lsi(corpus_tfidf, id2word=dictionary, num_topics=100)

# 用待检索的文档向量初始化一个相似度计算的对象
index = similarities.MatrixSimilarity(lsimodel[corpus_tfidf])

# 保存相似度矩阵，index中每一行表示一篇文档，列表示主题，列数与前面定义的num_topics数是一致的
index.save('./sim_mat.index')
index = similarities.MatrixSimilarity.load('./sim_mat.index')

# 计算一篇文档与现有语料中所有文档的（余弦）相似度
# 这里先取语料中的第一篇文档试验一下

query_bow = dictionary.doc2bow(clean_list[0])

# tfidf向量化
query_tfidf = tfidf[query_bow]

# 用之前训练好的LSI模型将其映射到topic空间
query_vec = lsimodel[query_tfidf]

# 检查query在index中的相似度
sim = index[query_vec]
print(list(enumerate(sim)))



