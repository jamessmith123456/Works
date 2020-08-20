from __future__ import print_function  #在开头加上from __future__ import print_function这句之后，即使在python2.X，使用print就得像python3.X那样加括号使用
import numpy as np
import os
import tensorflow as tf
from tqdm import tqdm
import _pickle as cPickle
import re
import itertools
from collections import Counter


PAD = "_PAD"
UNK = "_UNK"


def Q2B(uchar):
  """全角转半角"""
  inside_code = ord(uchar)  #ord()函数以一个字符（长度为1的字符串）作为参数，返回对应的 ASCII 数值，或者 Unicode 数值
  if inside_code == 0x3000: #空格的全角编码为0*3000，半角编码为0*0020
    inside_code = 0x0020
  else:
    inside_code -= 0xfee0 #其余字符，只要减去0*fee0即可从全角转换为半角
  #转完之后不是半角字符返回原来的字符
  if inside_code < 0x0020 or inside_code > 0x7e: #不属于全角范围的字符，保持原样
    return uchar
  return chr(inside_code)

  #unichr() 函数 和 chr()函数功能基本一样， 只不过是返回 unicode 的字符



def replace_all(repls, text):
  # return re.sub('|'.join(repls.keys()), lambda k: repls[k.group(0)], text)
  return re.sub('|'.join(re.escape(key) for key in repls.keys()), lambda k: repls[k.group(0)], text)


#re.sub(pattern, repl, string, count=0, flags=0)
#Return the string obtained by replacing the leftmost non-overlapping occurrences of pattern in string by the replacement repl.
#str.join(iterable)   简单来讲，就是将可迭代对象中的元素以 str 为分隔符拼接返回
#re.escape(pattern) 可以对字符串中所有可能被解释为正则运算符的字符进行转义的应用函数。如果字符串很长且包含很多特殊技字符，而你又不想输入一大堆反斜杠，
#或者字符串来自于用户(比如通过raw_input函数获取输入的内容)，且要用作正则表达式的一部分的时候，可以使用这个函数。

#dict.keys()
#keys() 函数以列表返回一个字典所有的键

#lambda 函数可以接收任意多个参数 (包括可选参数) 并且返回单个表达式的值

#一般，m.group(N) 返回第N组括号匹配的字符。
#而m.group() == m.group(0) == 所有匹配的字符，与括号无关，这个是API规定的

#split多个分隔符 单一分隔符，使用str.split()即可 
#多个分隔符，复杂的分隔情况，使用re.split


def split_sentence(txt):
  sents = re.split(r'\n|\s|;|；|。|，|\.|,|\?|\!|｜|[=]{2,}|[.]{3,}|[─]{2,}|[\-]{2,}|~|、|╱|∥', txt)
  sents = [c for s in sents for c in re.split(r'([^%]+[\d,.]+%)', s)]
  sents = list(filter(None, sents))
  return sents


#filter函数用于过滤序列。filter()函数包含2个参数，一个是过滤函数，另一个是过滤序列。形如：filter(f,L)
#filter函数会将过滤函数依次作用于每一个过滤序列的元素，返回符合要求的过滤值。


def normalize_punctuation(text):
  cpun = [['	'],
          ['﹗', '！'],
          ['“', '゛', '〃', '′', '＂'],
          ['”'],
          ['´', '‘', '’'],
          ['；', '﹔'],
          ['《', '〈', '＜'],
          ['》', '〉', '＞'],
          ['﹑'],
          ['【', '『', '〔', '﹝', '｢', '﹁'],
          ['】', '』', '〕', '﹞', '｣', '﹂'],
          ['（', '「'],
          ['）', '」'],
          ['﹖', '？'],
          ['︰', '﹕', '：'],
          ['・', '．', '·', '‧', '°'],
          ['●', '○', '▲', '◎', '◇', '■', '□', '※', '◆'],
          ['〜', '～', '∼'],
          ['︱', '│', '┼'],
          ['╱'],
          ['╲'],
          ['—', 'ー', '―', '‐', '−', '─', '﹣', '–', 'ㄧ', '－']]
  epun = [' ', '!', '"', '"', '\'', ';', '<', '>', '、', '[', ']', '(', ')', '?', ':', '･', '•', '~', '|', '/', '\\', '-']
  repls = {}

  for i in range(len(cpun)):
    for j in range(len(cpun[i])):
      repls[cpun[i][j]] = epun[i]

  return replace_all(repls, text)

#上面这个函数的作用：最终得到的repls应该是一个字典,字典的键是epun中的每一个标点符号，对应的
#键值是相应的标准符号




def clean_str(txt):
  # txt = txt.replace('説', '說')
  # txt = txt.replace('閲', '閱')
  # txt = txt.replace('脱', '脫')
  # txt = txt.replace('蜕', '蛻')
  # txt = txt.replace('户', '戶')
  # 臺
  txt = txt.replace('臺', '台')   #将txt里面的繁体字变为简体字
  txt = txt.replace('　', '') # \u3000 #将txt里面的空格变换掉
  txt = normalize_punctuation(txt) #将
  txt = ''.join([Q2B(c) for c in list(txt)])
  return txt


def build_vocab(sentences):
  """
  Builds a vocabulary mapping from word to index based on the sentences.
  Returns vocabulary mapping and inverse vocabulary mapping.
  """
  # Build vocabulary
  
  word_counts = Counter(itertools.chain(*sentences))
  # Mapping from index to word
  vocabulary_inv = [x[0] for x in word_counts.most_common()]
  # Mapping from word to index
  vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
  return [vocabulary, vocabulary_inv]


def get_vocab(path='./data/vocab.pkl'):
  """Loads the vocab file, if present"""
  if not os.path.exists(path) or os.path.isdir(path):
    raise ValueError('No file at {}'.format(path))

  char_list = cPickle.load(open(path, 'rb'))
  vocab = dict(zip(char_list, range(len(char_list))))

  return vocab, char_list


def build_dataset(pos_path='chinese/pos_t.txt', neg_path='chinese/neg_t.txt',
                  data_dir='./data', max_doc_len=30, max_sent_len=50, ):
  pos_docs = list(open(os.path.join(data_dir, pos_path)).readlines())
  neg_docs = list(open(os.path.join(data_dir, neg_path)).readlines())
  vocab, _ = get_vocab('./data/vocab.pkl')
  pos_size = len(pos_docs)
  neg_size = len(neg_docs)
  pos_train_size = int(pos_size * 0.9)
  pos_valid_size = pos_size - pos_train_size
  neg_train_size = int(neg_size * 0.9)
  neg_valid_size = neg_size - neg_train_size
  train_path = os.path.join(data_dir, 'train.tfrecords')
  valid_path = os.path.join(data_dir, 'valid.tfrecords')

  def upsampling(x, size):
    if len(x) > size:
      return x
    diff_size = size - len(x)
    return x + list(np.random.choice(x, diff_size, replace=False))


  def write_data(doc, label, out_f):
    doc = split_sentence(clean_str(doc))
    document_length = len(doc)
    sentence_lengths = np.zeros((max_doc_len,), dtype=np.int64)
    data = np.ones((max_doc_len * max_sent_len,), dtype=np.int64)
    doc_len = min(document_length, max_doc_len)

    for j in range(doc_len):
      sent = doc[j]
      actual_len = len(sent)
      pos = j * max_sent_len
      sent_len = min(actual_len, max_sent_len)
      # sentence_lengths
      sentence_lengths[j] = sent_len
      # dataset
      data[pos:pos+sent_len] = [vocab.get(sent[k], 0) for k in range(sent_len)]

    features = {'sentence_lengths': tf.train.Feature(int64_list=tf.train.Int64List(value=sentence_lengths)),
                'document_lengths': tf.train.Feature(int64_list=tf.train.Int64List(value=[doc_len])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                'text': tf.train.Feature(int64_list=tf.train.Int64List(value=data))}
    example = tf.train.Example(features=tf.train.Features(feature=features))
    out_f.write(example.SerializeToString())

  # oversampling
  with tf.python_io.TFRecordWriter(train_path) as out_f:
    train_size = max(pos_train_size, neg_train_size)
    pos_train_docs = np.random.choice(upsampling(pos_docs[:pos_train_size], train_size), train_size, replace=False)
    neg_train_docs = np.random.choice(upsampling(neg_docs[:neg_train_size], train_size), train_size, replace=False)

    print(len(pos_train_docs), len(neg_train_docs))
    for i in tqdm(range(train_size)):
      pos_row = pos_train_docs[i]
      neg_row = neg_train_docs[i]
      write_data(pos_row, 1, out_f)
      write_data(neg_row, 0, out_f)

  with tf.python_io.TFRecordWriter(valid_path) as out_f:
    valid_size = max(pos_valid_size, neg_valid_size)
    pos_valid_docs = np.random.choice(upsampling(pos_docs[pos_train_size:], valid_size), valid_size, replace=False)
    neg_valid_docs = np.random.choice(upsampling(neg_docs[neg_train_size:], valid_size), valid_size, replace=False)
    for i in tqdm(range(valid_size)):
      pos_row = pos_valid_docs[i]
      neg_row = neg_valid_docs[i]
      write_data(pos_row, 1, out_f)
      write_data(neg_row, 0, out_f)

  print('Done {} records, train {}, valid {}'.format(pos_size + neg_size,
                                                     pos_train_size + neg_train_size,
                                                     pos_valid_size + neg_valid_size))


if __name__ == '__main__':
  build_dataset()













#我自己整理的，太好了这个文章

import re

line = "word;Word;temp"
"""
单一分隔符，str.split()与 re.split()zuoy作用一致 
"""
print line.split("; ")
# ['word', 'Word', 'temp']
print re.split(r"; ", line)
# ['word', 'Word', 'temp']


"""
多个单一 分隔符 时 ，"[]"与 "|"的 效果是一样的，但是 请注意 使用 “|”时 mouxie某些字符 需要转义  
"""
line1 = "word;Word,emp?hahaha"
print re.split(r";|,|\?", line1) #别忘了转义"?"
# ['word', 'Word', 'emp', 'hahaha']
print re.split(r"[;,?]", line1)
# ['word', 'Word', 'emp', 'hahaha']

## 当 空格 出现 时 ，就 十分讨厌了(ノω<。)ノ))☆.。 
"""
多个 长短 不一 的的分隔符的分隔符时， 就应该使用 "|"
"""
line2 = "word;Word,emp? hahaha; whole, cai"
print re.split(r";|,|\?\s|;\s|,\s", line2) 
# ['word', 'Word', 'emp', 'hahaha', ' whole', ' cai']

## 以上 只是 为了 说明这些的吗的适用情况 ，还有 更加渐变简便 的 用法 是
print re.split(r"\W+", line)
print re.split(r"\W+", line1)
print re.split(r"\W+", line2)

"""
适用 “（）”则是 将 分隔 后的 结果 连同分隔符均 有所 保留 
"""
print re.split(r"(\W+)", line2)
# ['word', ';', 'Word', ',', 'emp', '? ', 'hahaha', '; ', 'whole', ', ', 'cai'] 
# 注意： 连 空格 都 保留了 
--------------------- 
作者：猪杂汤饭 
来源：CSDN 
原文：https://blog.csdn.net/programmer_at/article/details/77409507 
版权声明：本文为博主原创文章，转载请附上博文链接！
