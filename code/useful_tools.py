
import re
import numpy as np
import pandas as pd
import jieba
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
import itertools
import pickle 
from sklearn.model_selection import cross_val_score

stopword_list = [k.strip() for k in open('../data/stopword.txt', encoding='utf8').readlines() if k.strip() != '']
# 加入新的停用词
stop_word_new = ['一','二','三','四','五','六','七','八','九','十','不','未']
stopword_list.extend(stop_word_new)
stopword_list.remove('是')
stopword_list.remove('为了')

addword = ['为了','本规定所称', '是指','区块链','微博客','根服务器',
'提供者应当','鼓励','应当遵守','适用']
for i in addword:
    jieba.add_word(i)

def get_vectorize(wordlist,vector = 'CountVectorizer',feats =150):
    '''得到特征X矩阵
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
    '''jieba分词,正则替换数字
    '''
    string = re.sub("[0-9]"," ",string) # 正则替换数字
    cutWords = [k for k in jieba.cut(string) if k != '' if k not in stopword_list]
    combined = ' '.join(cutWords)
    return combined


def generate_ngrams(text, n_gram=2):
    '''得到N元组
    '''
    token = [k for k in jieba.cut(text) if k.isspace() == False if k != '' if k not in stopword_list]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    single = ['_'.join(j) for j in ngrams if j != '' if j not in stopword_list]
    if single != '_':
        combined = ' '.join(single)
    return combined

def plot_count(df,n):
    '''画条形图函数，df:数据框，n:前n个词
    '''
    uplot = pd.DataFrame(df.apply(lambda x: sum(x)),columns = ["Score"])
    uplot.sort_values(by = 'Score',inplace = True,ascending=False)
    uplot = uplot[0: n + 1]
    uplot.sort_values(by = 'Score',inplace = True,ascending=True)
    uplot.plot.barh(alpha=0.7,figsize=(6,8))
    
def plot_count_by(df, row, n,figsize=(10,12)):
    '''分面绘制条形图
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
    '''分数打印，可选交叉验证法或Bootstrap方法
    '''
    cvscore = cross_val_score(model, x, y, cv = cv)
    if isinstance(cv,int):
        print(cv,'折交叉验证准确率为',round(cvscore.mean(),3))
    else:
        print('Bootstrap抽样准确率为',round(cvscore.mean(),3))
    
def match_single(x, word):
    '''输入分词后的语料库和想匹配的单词，返回包含这个单词的对象
    '''
    if word in x:
        x = word
    else:
        x = ''
    return x

def add_single_col(og_matrix ,ogcolumn, new_word):
    '''增加新词列,返回增加后的DTM. 所有行都不包含这个词时返回原矩阵
    ''' 
    cutWords = ogcolumn.apply(lambda x: get_cutword(x))
    if new_word in og_matrix.columns:
        return og_matrix
    else:
        try:
            newwdvec = cutWords.apply(lambda x: match_single(x, new_word)) #得到想加入的列
            new_max = get_vectorize(newwdvec) # 得到单独一列DTM
            new_dtm = pd.concat([og_matrix, new_max], axis = 1, join='inner') 
            return new_dtm
        except ValueError:
            print('抱歉，无法搜索到这个词')
            return og_matrix
        

pkl_file = open('../classifier/colname.pkl','rb')    ## 以二进制方式打开文件
train_col = pickle.load(pkl_file) 
def consist_train_test(test):
    '''使得测试集列名顺序与训练集一致
    '''
    new_df = pd.DataFrame()
    for i in train_col:
        if i in test.columns:
            new_df[i] = test[i]
        else:
            new_df[i] = 0
    new_df.fillna(0, inplace = True)
    order = train_col
    new_df[order]
    return new_df

def add_words(mat, ogcolumn):
    '''用于预测新数据，集成了最终模型需要加入的四个词
    '''
    cidian = ['所称', '是指', '为了', '鼓励', '应当遵守']
    for i in cidian:
        mat = add_single_col(mat , ogcolumn, i)
    return mat