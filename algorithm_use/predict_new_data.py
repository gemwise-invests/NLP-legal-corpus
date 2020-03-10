

"""
调用训练好的模型预测新的数据集。训练好的算法存放在classifier文件夹内的pkl文件里。可以使用两种算法。
rf为随机森林，xgb为xgboost
本项目所有自编函数都存放在useful_tools.py内，使用时直接调用
新数据应该只有法律条文文本列，为test['content']
注意文件夹里所有的文件都不可以移动、重命名，否则会报错
"""
from useful_tools import * # 导入自编函数
from sklearn.externals import joblib


import os
os.getcwd()  # 返回目前所在的路径

#os.chdir('F:/algorithm_use') # 改路径！！！非常重要！！！如果你将algorithm_use放入F盘的话。必须改到predict_new_data.py文件所在的文件夹内！！！

xgb = joblib.load('./classifier/xgb.pkl')  # 导入训练好的2个算法
rffit = joblib.load('./classifier/rffit.pkl') 

#######################
test = pd.read_csv("./data/TestSet.csv", encoding = 'gb18030') # 读取需要预测的数据
#######################

# 将test['content'] 转化为自变量词频矩阵(Document-Term Matrix)
cutWords = test['content'].apply(lambda x: get_cutword(x)) # 得到的是pandas series
unigram_test = get_vectorize(cutWords, feats = 100) # 将初始变量数设为100
unigram_test = add_words(unigram_test ,test['content']) # 加入训练时的少数关键词

newunigram = consist_train_test(unigram_test) # newunigram 为我们用于放入算法的X矩阵

# 分别用两个算法进行预测
y_rf = rffit.predict(newunigram) # 随机森林
y_xgb = xgb.predict(newunigram) # Xgboost

# 建立字典将数值映射为法律条文分类
recode =  {0 : '使用者要求', 1 : '名词解释',2 : '服务监督', 3 : '法规倡议', 
           4 : '法规目的',5 : '职责区分', 6 : '运营者要求', 7 : '违规处理'}
# 映射转换
y_rf = pd.Series(y_rf).map(recode)
y_xgb = pd.Series(y_xgb).map(recode)

# 得到加入预测结果列的数据框
predicted_rf = pd.DataFrame({'Content': test['content'], 'class': y_rf })
predicted_xgb = pd.DataFrame({'Content': test['content'], 'class': y_xgb })

# 数据框保存为CSV文件，这个就是包含了预测值列的文件
predicted_rf.to_csv('./data/predicted_rf.csv',encoding = 'gb18030',index = False)
predicted_xgb.to_csv('./data/predicted_xgb.csv',encoding = 'gb18030',index = False)
