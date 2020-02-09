# -*- coding: utf-8 -*-
"""
@ Machine Learning Pipeline
@author: xsong
"""
from sklearn.model_selection import cross_val_score
#from sklearn.naive_bayes import GaussianNB
#from sklearn.naive_bayes import MultinomialNB
#from sklearn.naive_bayes import BernoulliNB

def hello():
    print('say hello')
    
    
def cross_print_info(model,x,y,cv):
    '''
    交叉验证分数打印
    '''
    cvscore = cross_val_score(model, x, y, cv=cv)
    print('10折交叉验证准确率为',round(cvscore.mean(),3))
    
