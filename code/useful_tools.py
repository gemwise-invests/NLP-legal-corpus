
import re
import numpy as np
import pandas as pd
import jieba
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
import itertools
from sklearn.model_selection import cross_val_score

stopword_list = [k.strip() for k in open('../data/stopword.txt', encoding='utf8').readlines() if k.strip() != '']
# 加入新的停用词
stop_word_new = ['一','二','三','四','五','六','七','八','九','十','不','未','互联网','网络','服务']
stopword_list.extend(stop_word_new)
jieba.add_word('区块链')
jieba.add_word('微博客')
jieba.add_word('根服务器')
jieba.del_word('监督管理')

def get_vectorize(wordlist,vector = 'CountVectorizer',feats =150):
    '''
    得到特征X矩阵
    '''
    if vector == 'CountVectorizer':
        cv = CountVectorizer(max_features = feats,token_pattern='[\u4e00-\u9fa5_a-zA-Z0-9]{1,}')
        cv_fit = cv.fit_transform(wordlist).toarray()
        colnames = cv.get_feature_names()
        word_matrix = pd.DataFrame(cv_fit, columns=colnames)   
        return word_matrix
    elif vector == 'TfidfVectorizer':
        cv = TfidfVectorizer(max_features = feats,token_pattern='[\u4e00-\u9fa5_a-zA-Z0-9]{1,}')
        cv_fit = cv.fit_transform(wordlist).toarray()
        colnames = cv.get_feature_names()
        word_matrix = pd.DataFrame(cv_fit, columns=colnames)   
        return word_matrix
    elif vector == 'HashingVectorizer':
        cv =  HashingVectorizer(n_features = feats)
        cv_fit = cv.fit_transform(wordlist).toarray()
        word_matrix = pd.DataFrame(cv_fit)
        return word_matrix
        
    

def get_cutword(string):
    '''
    jieba分词,正则替换数字
    '''
    string = re.sub("[0-9]"," ",string) # 正则替换数字
    cutWords = [k for k in jieba.cut(string) if k != '' if k not in stopword_list]
    combined = ' '.join(cutWords)
    return combined


def generate_ngrams(text, n_gram=2):
    '''
    得到N元组
    '''
    token = [k for k in jieba.cut(text) if k.isspace() == False if k != '' if k not in stopword_list]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    single = ['_'.join(j) for j in ngrams if j != '' if j not in stopword_list]
    if single != '_':
        combined = ' '.join(single)
    return combined

def plot_count(df,n):
    '''
    画条形图函数，df:数据框，n:前n个词
    '''
    uplot = pd.DataFrame(df.apply(lambda x: sum(x)),columns = ["Score"])
    uplot.sort_values(by = 'Score',inplace = True,ascending=False)
    uplot = uplot[0: n + 1]
    uplot.sort_values(by = 'Score',inplace = True,ascending=True)
    uplot.plot.barh(alpha=0.7,figsize=(6,8))
    
def plot_count_by(df, row, n,figsize=(10,12)):
    '''
    分面绘制条形图
    '''
    uplot = pd.Series(df.iloc[row,])
    uplot.sort_values(inplace = True,ascending=False)
    uplot = uplot[0: n + 1]
    uplot.sort_values(inplace = True,ascending=True)
    uplot.plot.barh(alpha=0.7,figsize=figsize)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    
def cross_print_info(model,x,y,cv): # 模型, x, y, 交叉验证数
    '''
    交叉验证分数打印
    '''
    cvscore = cross_val_score(model, x, y, cv=cv)
    print(cv,'折交叉验证准确率为',round(cvscore.mean(),3))
    
