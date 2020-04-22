# -*- coding:utf-8 -*-
'''
主题: 法律文本分类程序
作者：宋骁
日期：2020-04-17
Python版本: 3.7.3

Title: Law Text Classification Script
Author: Xiao Song 
Date：2020-04-17
Python-Version: 3.7.3

调用训练好的模型预测新的数据集。训练好的算法存放在classifier文件夹内的pkl文件里。可以使用两种算法。
rf为随机森林，xgb为xgboost
新数据应该只有法律条文文本列，为test['content']
注意文件夹里所有的文件都不可以移动、重命名，否则会报错
'''

from sklearn.externals import joblib # 用于加载分类器
from optparse import OptionParser # 命令行参数
import re
import numpy as np
import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer
import itertools
import pickle 
import sys

addword = ['为了','本规定所称', '是指','区块链','微博客','根服务器',
'提供者应当','鼓励','应当遵守','适用']
for i in addword:
    jieba.add_word(i)

def get_cutword(string):
    '''jieba分词,使用正则表达式替换数字
    '''
    with open('./data/stopword_list.pkl', 'rb') as f: #加载停用词
        stopword_list = pickle.load(f) 
    string = re.sub("[0-9]"," ",string) # 正则替换数字
    cutWords = [k for k in jieba.cut(string) if k != '' if k not in stopword_list]
    combined = ' '.join(cutWords)
    return combined

def get_vectorize(wordlist,vector = 'CountVectorizer',feats = 150):
    '''得到特征X矩阵
    '''
    if vector == 'CountVectorizer':
        cv = CountVectorizer(max_features = feats,token_pattern='[\u4e00-\u9fa5_a-zA-Z0-9]{1,}')
    elif vector == 'TfidfVectorizer':
        cv = TfidfVectorizer(max_features = feats,token_pattern='[\u4e00-\u9fa5_a-zA-Z0-9]{1,}')

    cv_fit = cv.fit_transform(wordlist).toarray()
    colnames = cv.get_feature_names()
    word_matrix = pd.DataFrame(cv_fit, columns=colnames)   
    return word_matrix

def consist_train_test(test):
    '''使得测试集列名顺序与训练集一致
    '''
    with open('./data/colname.pkl', 'rb') as u: # 加载训练集列名
        train_col = pickle.load(u) 
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

def add_words(mat, ogcolumn):
    '''用于预测新数据，集成了最终模型需要加入的四个词
    '''
    cidian = ['所称', '是指', '为了', '鼓励', '应当遵守']
    for i in cidian:
        mat = add_single_col(mat , ogcolumn, i)
    return mat

def dtm_convert(col):
    '''转化为特征矩阵
    '''
    try:
        cutWords = test[col].apply(lambda x: get_cutword(x)) # 得到的是pandas series
    except KeyError:
        print('未找到名为\'%s \'的数据字段，请检查数据'%col)
        sys.exit()
    unigram_test = get_vectorize(cutWords, feats = 100) # 将初始变量数设为100
    unigram_test = add_words(unigram_test ,test[col]) # 加入训练时的少数关键词
    newunigram = consist_train_test(unigram_test) # newunigram 为我们用于放入算法的X矩阵
    print('文本矩阵化完毕')
    return newunigram

def load_classifer(cls):
    '''加载分类器
    '''
    clss = {'xgb': './classifier/xgb.pkl', 
             'rf': './classifier/rffit.pkl'}
    loaded_cls = joblib.load(clss[cls]) 
    print('分类算法加载完毕')
    return loaded_cls

def predict_result(loaded_cls,newunigram):
    '''预测结果
    '''
    # 建立字典
    recode =  {0 : '使用者要求', 1 : '名词解释',2 : '服务监督', 3 : '法规倡议', 
           4 : '法规目的',5 : '职责区分', 6 : '运营者要求', 7 : '违规处理'}
    y_pred = loaded_cls.predict(newunigram)
    y_pred = pd.Series(y_pred).map(recode) # 映射转换
    print('数据拟合完毕')
    return y_pred

if __name__ == "__main__": # 作为脚本时运行

    parser = OptionParser(usage='法律文本分类命令行程序',prog = 'lawPredict',
                            description = 'lawPredict是一个Python命令行程序，输入待分类的csv数据文件，输出分类结果' ) # 设置命令行参数

    parser.add_option('-i', '--input', dest='input_file', type='string', default='./data/TestSet.csv',
                      help = '输入文件名，input file name') 

    parser.add_option('-o', '--output', dest='output_file', type='string', default='./data/result.csv',
                      help = '输出文件名， output file name')

    parser.add_option('-c', '--classifier', dest='classifier', type='string', default='xgb', 
                      help = '机器学习分类算法，xgb:xgboost; rf:随机森林') 

    parser.add_option('-q', action = 'store_false', dest = 'include_og', default = True,
                      help = '输出文件是否包含原始文本列，默认包含') 
    
    parser.add_option('-n','--column', dest='column_name', type='string', default='content',
                      help = '输入文本列字段名，默认为content') 


    options, args = parser.parse_args()
    input_path = options.input_file # 输入文件 file input
    out_path = options.output_file # 输出 file output
    cls_name = options.classifier # 分类器 ML classifer
    include_og = options.include_og # 是否包含原始文本列 if include origin text column(content)
    column_name = options.column_name # 列名 column name
    try:
        test = pd.read_csv(input_path, encoding = 'gb18030') # 读取需要预测的数据
        print('数据读取完毕')
    except FileNotFoundError:
        print('未找到名为%s的测试集数据，请检查data文件夹'%input_path)
        sys.exit()

    try:
        loaded_cls_name = load_classifer(cls_name) # 加载分类器
    except KeyError:
        print('不存在名为%s的分类器，请重新输入'%cls_name)
        sys.exit()

        
    newunigram = dtm_convert(column_name)
    y_pred = predict_result(loaded_cls_name, newunigram)
    
    if include_og:
        predicted_df = pd.DataFrame({'content': test[column_name], 'class': y_pred })
    else:
        predicted_df = pd.DataFrame({'class': y_pred })

    predicted_df.to_csv(out_path,encoding = 'gb18030', index = False)
    print('保存输出文件完毕，文件位于%s'%out_path)











