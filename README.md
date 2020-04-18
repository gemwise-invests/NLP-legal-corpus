
# law_learning


![](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-brightgreen.svg)


作者：[宋骁](https://xsong.ltd/)  

[项目主页](https://programfun.netlify.com/laws/)  

[最终程序](https://github.com/songxxiao/law_learning/releases)

## 特性

+ 使用Jupyter Notebook展示算法训练过程和数据分析过程

+ 使用命令行程序封装训练好的算法，便于重复使用


## 初始化

```
git clone git@github.com:songxxiao/law_learning.git
```
需要的扩展库

```
pip install jieba
pip install scikit-learn
pip install numpy
pip install pandas
```
Python版本: 3.7.3

## 使用方法

本程序分为两部分，其一是保存在`./code`文件夹和[项目主页](https://programfun.netlify.com/laws/)中模型训练的jupyter notebooks；另一个是划分法律条文类别的Python命令行程序，程序脚本文件为`./lawPredict/lawPredict.py` 。下面主要介绍命令行程序的使用方法。

如果想预测你的法律文本，需要将你的法律条文放在一个.csv文件里，每一行代表一条待分类的法律。(什么？！我只知道.xlsx格式！那需要在Excel中依次点击文件-另存为-.csv文件)。然后将这个csv放入./data文件夹中。

想成功运行`lawPredict.py`，你最好已经熟悉Linux风格的命令行，如`git commit -m 'message'`之类。此命令为使用`git` 进行`commit`操作，`-m`是参数`--message`的简写，它后面紧跟着想要传递的字符串。这个程序同样使用这种方式传递参数。从帮助中可以看出参数共有以下几个：

```
$ python lawPredict.py --help
Usage: 法律文本分类命令行程序

lawPredict是一个Python命令行程序，输入待分类的csv数据文件，输出分类结果

Options:
  -h, --help            show this help message and exit
  -i INPUT_FILE, --input=INPUT_FILE
                        输入文件名，input file name
  -o OUTPUT_FILE, --output=OUTPUT_FILE
                        输出文件名， output file name
  -c CLASSIFIER, --classifier=CLASSIFIER
                        机器学习分类算法，xgb:xgboost; rf:随机森林
  -q                    输出文件是否包含原始文本列，默认包含
  -n COLUMN_NAME, --column=COLUMN_NAME
                        输入文本列字段名，默认为content
```

+ -i 指定你输入文件的路径，默认为'./data/TestSet.csv',也就是你需要将你的测试集放到`./lawPredict/data`文件夹里

+ -o 指定输出文件的路径，默认为'./data/result.csv',带有预测值 csv文件将会出现在`./lawPredict/data`文件夹里

+ -c 使用哪种算法进行预测？目前提供了Xgboost和随机森林两种训练好的算法

+ -q 布尔参数，用于控制输出文件是否包含你需要分类的文本。包含的话相当于在原始的csv中多增加一列预测值

+ -n 文本字段名，如果输入文件中有多列的话，用于指定输入文件的文本列，默认为`content`

默认参数如下：

```
$ python lawPredict.py -i './data/TestSet.csv' -o './data/result.csv' -c 'xgb' -n 'content'
```

如果中途任意一行运行不出来，十有八九是因为你的路径没改对或是扩展库没安装。你**必须得把路径改到`lawPredict.py` 文件所在的`lawPredict`文件夹中**。

```
$ cd './lawPredict'
$ python lawPredict.py -c 'rf'
```

只有改对了路径，文件内部的形如'./data/TestSet.csv'这样的相对路径才能起作用。
如果你用的是Python自带的IDLE，那么你需要先确保安装了所有必备的扩展库。安装的方法是各种`pip install`或`conda install`。

如果读不懂代码，可以看看注释。命令行工具可以使用Anaconda Prompt 或Jupyter 的Terminal；Linux使用Terminal。根据我的测试，Ubuntu Terminal需要使用python3命令：

```
$ python3 lawPredict.py -c 'rf'
```

运行成功的标志是你的data文件夹中多出来1个.csv文件，`result.csv`。这个就是包含了预测值列的文件了。

## 文件说明

| lawPredict文件夹  | 说明                                                                                    |
|--------------------------|-----------------------------------------------------------------------------------------|
| ./data/stopword_list.pkl | 停用词表                                                                                |
| ./data/TestSet.csv       | 待预测的新数据，content列为法律文本                                                     |
| ./classifier/colname.pkl | 储存了训练集清理好X矩阵的列名list，用于规范测试集列名。训练集和测试集列名必须保持一致。 |
| ./classifier/rffit.pkl   | 训练好的随机森林模型，直接用于预测。                                                    |
| ./classifier/xgb.pkl     | 训练好的Xgboost模型，直接用于预测。                                                     |
| ./lawPredict.py          | 预测新数据的Python命令行程序                                                            |