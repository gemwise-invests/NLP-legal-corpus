# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 15:03:27 2020

@author: xsong
"""

# -*- coding: UTF8 -*-

import pickle
import io
import sys
import json
import re
import xlrd
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import jieba
from sklearn.linear_model import SGDClassifier

sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer, encoding='gb18030')  # 改变标准输出的默认编码

# 重点是rb和r的区别，rb是打开2进制文件，文本文件用r
f = open('Lawlist.pkl', 'rb')
lawlist = pickle.load(f)
# print(lawlist)
lawarticle = []
for list_num in range(len(lawlist)):
    lawarticle.append(lawlist[list_num][4])
# print(lawarticle)
'''
file = open('E:/文档/团队/任务/项目/原始文本的分类/关键词json/332732.json', "rb")
fileJson = json.load(file)
print(fileJson, len(fileJson))
'''
lawpiece = []
for i in range(len(lawarticle)):
    lawpiece.append(lawarticle[i].split('\n\u3000\u3000'))  # 分开每一条
# print(lawpiece)

for i in range(len(lawpiece)):
    for m in range(len(lawpiece[i])):
        lawpiece[i][m] = lawpiece[i][m][:lawpiece[i][m].find('\n\n\n\n')]
        lawpiece[i][m] = lawpiece[i][m].strip()
        lawpiece[i][m] = lawpiece[i][m].replace('\u3000', '')
        lawpiece[i][m] = lawpiece[i][m].replace('\n', '')
        lawpiece[i][m] = lawpiece[i][m].replace('\r', '')
        lawpiece[i][m] = lawpiece[i][m].replace(' ', '')
law = []
for i in range(len(lawpiece)):
    for m in lawpiece[i]:
        if '\xa0' not in m and '│' not in m and '。' in m:
            law.append(m)
law = law[:len(lawpiece)]
for i in range(len(law)):
    if law[i].find('（') == 0:
        law[i] = ''
    elif law[i].find('【') == 0:
        index1 = law[i].find('】')
        law[i] = law[i][index1 + 1:]
    elif law[i].find('【') != 0 and law[i].find('【') != -1:
        law[i] = law[i].replace('【', '')
        law[i] = law[i].replace('】', '')
    elif law[i].find('附：') != -1:
        index2 = law[i].find('附：')
        law[i] = law[i][:index2]
    elif law[i].find('附录一：') != -1:
        index3 = law[i].find('附录一：')
        law[i] = law[i][:index3]
    elif law[i].find('附表：') != -1:
        index4 = law[i].find('附表：')
        law[i] = law[i][:index4]
total_law = []
for i in law:
    if '。' in i:
        total_law.append(i)
# print(len(total_law))

# 导入训练集
workbook = xlrd.open_workbook("E:\文档\团队\任务\项目\原始文本的分类\标注数据集.xlsx")
sheet1_object = workbook.sheet_by_name('Sheet1')
train_datalist = sheet1_object.col_values(0)
train_labelist = sheet1_object.col_values(1)
#print(train_datalist, train_labelist)
for i in range(len(train_labelist)):  # 标签转数值
    if train_labelist[i] != '惩罚':
        train_labelist[i] = -1
    else:
        train_labelist[i] = 1
for i in range(len(train_datalist)):
    train_datalist[i] = train_datalist[i].replace('\n', '')
    train_datalist[i] = train_datalist[i].replace('\u3000', '')
#print(train_datalist, train_labelist)


# 加载停用词表
def get_stopword_list():
    stop_word_path = r'E:\文档\团队\任务\项目\原始文本的分类\stopword.txt'
    stopword_list = [sw.replace('\n', '') for sw in open(
        stop_word_path, encoding='utf-8').readlines()]
    return stopword_list

# readline和readlines,一个返回字符串，一个返回列表。


# 去除停用词
def word_filter(seg_list):
    stopword_list = get_stopword_list()
    filter_list = []
    for seg in seg_list:
        if seg not in stopword_list and len(seg) > 1:
            if re.search('[0-9]', seg):
                continue
            else:
                filter_list.append(seg)
    return filter_list


# 分词接口
def seg_to_list(sentence):
    seg_list = jieba.cut(sentence)
    return seg_list
# 加载数据集，去停用词


def load_data(corpus_list):
    doc_list = []
    for content in corpus_list:
        seg_list = seg_to_list(content)
        filter_list = word_filter(seg_list)
        content = ' '.join(filter_list)
        doc_list.append(content)  # 列表里嵌套列表。
    return doc_list
# 文本向量化


def countvec(doc_list):

    # 中文特征值化

    cv = CountVectorizer()  # 返回sparse矩阵

    # 调用fit_transform  : 这就是要把数据输入进去了。
    cv_fit = cv.fit_transform(doc_list)  # 包含字符串的列表！！！

    # print(cv_fit.get_feature_names())  #实例化里边的方法。统计所有文章中的所有的词，重复的只看作一次。把每一篇文章对应的八个词标记，统计次数,不再是虚拟变量。单个字母不统计。
    print(cv.vocabulary_)
    print(cv_fit.toarray())  # sparse矩阵转化成数组形式。
    return cv_fit.toarray(), cv.vocabulary_


get_stopword_list()


if __name__ == "__main__":
    train_X, train_topic_word = countvec(load_data(corpus_list=train_datalist))
    test_X, test_topic_word = countvec(load_data(corpus_list=total_law))
    # print(train_X.sum())

# 输出矩阵X
train_datamat = pd.DataFrame(train_X)  # (214, 1104)
train_datamat.to_excel("train矩阵.xlsx")
test_datamat = pd.DataFrame(test_X)  # (1171, 3697)
test_datamat.to_excel("test矩阵.xlsx")
print(train_datamat.shape, test_datamat.shape)
# SVM


def train_predict_evaluate_model(classifier, train_features, train_labels, test_features):
    classifier.fit(train_features, train_labels)  # 建模
    predictions = classifier.predict(test_features)  # 预测
    return predictions


svm = SGDClassifier(loss='hinge')

svm_bow_predictions = train_predict_evaluate_model(
    classifier=svm, train_features=train_datamat, train_labels=np.array(train_labelist), test_features=test_datamat)
print(svm_bow_predictions)