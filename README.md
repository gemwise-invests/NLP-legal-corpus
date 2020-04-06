# 使用方法

[宋骁](https://xsong.ltd/)  

[项目主页](https://programfun.netlify.com/laws/)  

[最终程序](https://github.com/songxxiao/law_learning/releases)



#### 注意文件夹里所有的文件都不可以移动、重命名，否则会报错！！

这是一个用于划分法律条文类别的程序，程序位于`predict_new_data.py` 内。如果想预测你的法律文本，需要将你的法律条文放在一个.csv文件里，这个文件最好只有一列，列名为`content`。每一行代表一条待分类的法律。(什么？！我只知道.xlsx格式！那请在Excel中依次点击文件-另存为-.csv文件)。然后将这个csv放入./data文件夹中(重命名为TestSet.csv并替换原有的TestSet.csv)。


想运行`predict_new_data.py` 有许多办法。一般还是手动运行比较好，首先在任何一个Python IDE中打开`predict_new_data.py`。 然后找到`os.chdir('F:/algorithm_use')`那一行改路径，之后读入数据依次运行每一行就对了。

有一个问题是如果
```
test = pd.read_csv("./data/TestSet.csv", encoding = 'gb18030')
```
读入的数据框是乱码的话，可以删除`encoding`参数，改为
```
test = pd.read_csv("./data/TestSet.csv")
```


如果中途任意一行运行不出来，十有八九是因为你的路径没改对或是扩展库没安装。前提你**必须得把路径改到`predict_new_data.py` 文件所在的`algorithm_use`文件夹中**。

如果你不知道你的路径可以使用如下的命令返回：
```
import os
os.getcwd()   
```
改路径的方法是（假设你把`algorithm_use`文件夹放到了F盘）：
```
os.chdir('F:/algorithm_use') 
```
只有改对了路径，文件内部的形如'./data/TestSet.csv'这样的相对路径才能起作用。
如果你用的是Python自带的IDLE，那么你需要先确保安装了所有必备的扩展库。安装的方法是各种`pip install`或`conda install`。

之后顺次根据代码顺序运行就可以了。如果读不懂代码，可以看看注释。   
以上是手动运行的方法。

如果想把`predict_new_data.py`当做一个完整的程序运行，可以使用Windows的CMD（Linux使用Terminal）。但需要将你的Python或IPython内核放到环境变量中去。


或者更方便的是使用Anaconda PowerShell Prompt（这样无需设置什么环境变量了），


然后依然是改路径，运行整个.py文件：

```
cd 'F:/algorithm_use' # 改路径到predict_new_data.py所在的文件夹
predict_new_data.py
```

运行成功的标志是你的data文件夹中多出来2个.csv文件：predicted_rf.csv和predicted_xgb.csv。这个就是包含了预测值列的文件了。

<table style="border-collapse:collapse;border-spacing:0" class="tg"><tr><th style="font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#656565;background-color:#efefef;text-align:left;vertical-align:middle">文件名</th><th style="font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#656565;background-color:#efefef;text-align:left;vertical-align:middle">说明</th></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#656565;background-color:#efefef;text-align:left;vertical-align:middle">./data/stopword.txt</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#656565;background-color:#efefef;text-align:left;vertical-align:middle">停用词表</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#656565;background-color:#efefef;text-align:left;vertical-align:middle">./data/TestSet.csv</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#656565;background-color:#efefef;text-align:left;vertical-align:middle">待预测的新数据，content列为法律文本，任何想通过此程序进行分类的数据都需要重命名为TestSet.csv</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#656565;background-color:#efefef;text-align:left;vertical-align:top">./classifier/colname.pkl</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#656565;background-color:#efefef;text-align:left;vertical-align:top">储存了训练集清理好X矩阵的列名list，用于规范测试集列名。训练集和测试集列名必须保持一致。</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#656565;background-color:#efefef;text-align:left;vertical-align:top">./classifier/rffit.pkl</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#656565;background-color:#efefef;text-align:left;vertical-align:top">训练好的随机森林模型，直接用于预测。</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#656565;background-color:#efefef;text-align:left;vertical-align:top">./classifier/xgb.pkl</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#656565;background-color:#efefef;text-align:left;vertical-align:top">训练好的Xgboost模型，直接用于预测。</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#656565;background-color:#efefef;text-align:left;vertical-align:top">./useful_tools.py</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#656565;background-color:#efefef;text-align:left;vertical-align:top">训练和预测过程中需要的所有自定义函数。</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#656565;background-color:#efefef;text-align:left;vertical-align:top">./predict_new_data.py</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#656565;background-color:#efefef;text-align:left;vertical-align:top">预测新数据的程序。可以手动也可以通过CMD自动执行。</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#656565;background-color:#efefef;text-align:left;vertical-align:top">./程序说明.html</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#656565;background-color:#efefef;text-align:left;vertical-align:top">本文件，说明怎么运行程序。</td></tr></table>