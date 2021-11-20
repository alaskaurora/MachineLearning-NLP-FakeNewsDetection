
import numpy as np
import pandas as pd
import sklearn as sk
import csv
import jieba
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import  CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
#创造停用词表，将文件夹中词表文件合并成一个
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
#利用中文分词开源库将新闻分词
def split(sentence):
    from re import match
    words = jieba.lcut(sentence,HMM = False)
    words = [x.strip() for x in words]
    #不能在停用词中，去除数字
    return list(filter(lambda x: x not in stopwords and match("\d+$",x) == None,words))

#利用正则表达式将内容中与文本无关如表情等元素删除
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
    comment = re.sub(r"(回复)?(//)?\s*@\S*?\s*(:| |$)", " ", comment)  # 去除正文中的@和回复/转发中的用户名
    comment = re.sub(r"\[\S+\]", "", comment)      # 去除表情符号
    comment = re.sub(r"#\S+#", "", comment)      # 保留话题内容
    comment = comment.replace("转发微博", "")       # 去除无意义的词语
    comment = re.sub(r"\s+", " ", comment) # 合并正文中过多的空格
    return comment
#读取文件
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
#将为填充的nan补充为’‘
train_data = train_data.fillna('')
test_data = test_data.fillna('')

#将评论与新闻合并，提高模型的精度
train_data['merge'] = train_data['content']+train_data['comment_all']
test_data['merge'] = test_data['content']+test_data['comment_all']
#数据清洗
stopwords = makeStopWord()

train_data['cut_merge'] = train_data['merge'].apply(cleanComment)
test_data['cut_merge'] = test_data['merge'].apply(cleanComment)

train_data['cut_merge'] = train_data['merge'].apply(split)
test_data['cut_merge'] = test_data['merge'].apply(split)
#将分割完的保存
train_data['cut_merge'].to_csv('train_cut.csv')
test_data['cut_merge'].to_csv('test_cut.csv')


X_train = train_data['cut_merge']
Y_train = train_data['label']

X_test = test_data['cut_merge']

#利用词包模型将词库转为词向量
Vectorizer = CountVectorizer(max_df = 0.8,
                            min_df = 3,
                            token_pattern = u'(?u)\\b[^\\d\\W]\\w+\\b',
                            stop_words =stopwords )
#选用朴素贝叶斯进行训练
nb = MultinomialNB()

X_train_vect =  Vectorizer.fit_transform((" ".join(x) for x in train_data['cut_merge'].values))

nb.fit(X_train_vect, Y_train)

X_vec = Vectorizer.transform((" ".join(x) for x in test_data['cut_merge'].values))

nb_result = nb.predict(X_vec)
#将训练完数据保存到result.txt文件中
np.savetxt('result.txt',np.array(nb_result),fmt = '%d')



