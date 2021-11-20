
import numpy as np
import pandas as pd
import sklearn as sk
import csv
import jieba
import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

train_data = csv.reader(open('train.csv'))
#合并一些停用词库成为总停用词库
def makeStopWord():
    from os import listdir
    stopword = set()
    for file in listdir('stopwords'):
        if file.endswith("txt"):
            with open(f'stopwords/{file}')as f:
                lines = f.readlines()
                for line in lines:
                    words = jieba.lcut(line,cut_all = False)
                    for word in words:
                        stopword.add(word.strip())
    return stopword

#停用词库
stopwords = makeStopWord()

def split_with_jieba(sentence):
    from re import match
    words = jieba.lcut(sentence,HMM = False)
    words = [x.strip() for x in words]
    #不能在停用词中，去除数字
    return list(filter(lambda x: x not in stopwords and match("\d+$",x) == None,words))

def cleanComment(comment):
    import re
    comment = re.sub('#.*?#', '', comment)
    comment = re.sub('//@.*?:', '', comment)
    comment = re.sub('//@.*?：', '', comment)
    comment = re.sub('//.*?:', '', comment)
    comment = re.sub('//.*?：', '', comment)
    comment = re.sub('【.*?】', '', comment)
    comment = re.sub('《.*?》', '', comment)
    comment = re.sub('//.*?//', '', comment)
    comment = re.sub('@.*?：', '', comment)
    comment = re.sub('@.*?:', '', comment)
    comment = re.sub('『.*?』', '', comment)
    comment = re.sub(r'\d', '', comment)
    return comment

 ##载入分词好的csv文件
train_df = pd.read_csv("./train_jieba.csv")
test_df = pd.read_csv("./test_labeld_jieba.csv")
train_df['jieba_cut'] = train_df['jieba_cut'].apply(eval)
test_df['jieba_cut'] = test_df["jieba_cut"].apply(eval)

df = pd.concat([train_df,test_df])



