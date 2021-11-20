
import numpy as np
import pandas as pd
import sklearn as sk
import csv
import jieba
import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

train_data = pd.read_csv('train.csv')

train_data.head(5)