#%%
from IPython.display import Latex
Latex('''

朴素贝叶斯算法
【朴素：特征条件独立   贝叶斯：基于贝叶斯定理】

1朴素贝叶斯的概念【联合概率分布、先验概率、条件概率**、全概率公式】【条件独立性假设、】   极大似然估计

2优缺点    
【优点： 分类效率稳定；对缺失数据不敏感，算法比较简单，常用于文本分类；在属性相关性较小时，该算法性能最好    缺点：假设属性之间相互独立；先验概率多取决于假设；对输入数据的表达形式很敏感】

3先验概率、后验概率
先验概率的计算比较简单，没有使用贝叶斯公式；
而后验概率的计算，要使用贝叶斯公式，而且在利用样本资料计算逻辑概率时，还要使用理论概率分布，需要更多的数理统计知识。

4朴素贝叶斯的参数估计：
①极大似然估计（可能出现概率为0的情况）②贝叶斯估计（加入常数，拉普拉斯平滑）


$p(A)$ ：事件A发生的概率； 
$p(A\cap\;B)$ :事件A 和事件B同时发生的概率 
$p(A|B)$ ：表示事件A在事件B发生的条件下发生的概率，
''')

#%%


import pandas as pd
import numpy as np
from IPython.display import Latex
import os



class NaiveBayes(object):
    #返回数据表和标签
    def getTrainSet(self):
        
        # __file__=os.path.realpath("Bayes.py")+"naivebayes_data.csv"
        # print(os.path.realpath("Bayes.py"))
        dataSet = pd.read_csv(__file__)
        # dataSet = pd.read_csv("C:\Users\Magneto_Wang\Documents\GitHub\机器学习个人笔记\机器学习实战\模式识别课程\贝叶斯估计\naivebayes_data.csv")
        dataSetNP = np.array(dataSet)

        #除了最后一列不取，前面列数全取
        trainData = dataSetNP[:,0:dataSetNP.shape[1]-1]

        #没有冒号表示取当前列，也就是最后一列
        labels = dataSetNP[:,dataSetNP.shape[1]-1]
        return trainData,labels
    
    #求labels 中的每个label的先验概率
    def classify(self,trainData,labels,features):
        
        
        labels =list(labels)
        probability_y={}#概率集合
        

        Latex('''  $p(A)$  ：事件A发生的概率  ''')
        #每一个标签数量除以总数
        for label in labels:
            probability_y[label] = labels.count(label)/float(len(labels))

        Latex('''  $p(A\cap\;B)$  ：事件A 和事件B同时发生的概率   ''')
        #求label与feature同时发生的概率
        probability_xy={}
        for y in probability_y.keys():
            #labels 放入枚举函数中，形成（序号，值）这一数据结构
            y_index = [i for i,label in enumerate(labels) if label == y]
            #y_index 对应标签的一行值 的 所有位置

            for j in range(len(features)):
                x_index = [i for i,feature in enumerate(trainData[:,j]) if feature == features[j]]
                #set 是集合，& 交集，| 并集，- 差集 ，in 属于,not in 不属于
                #求 x_index 和 y_index的 两个集合的交集。也就是同时发生的次数
                xy_count = len(set(x_index)&set(y_index))
                

                probaility_key = str(features[j])+'*'+str(y)
                print(probaility_key)
                probability_xy[probaility_key] = xy_count/float(len(labels))

        #条件概率
        Latex('''  $p(A|B)$  ：表示事件A在事件B发生的条件下发生的概率  ''')
         #P[X1/Y] = P[X1Y]/P[Y]
        Latex(''' $P[X|Y] = P[X\cap\;Y]/P[Y] $''')
        probability={}
        for y in probability_y.keys():
            for x in features:
                pkey = str(x)+'|'+str(y)

                
                probability[pkey] = probability_xy[str(x)+'*'+str(y)]/float(probability_y[y])
        

        #求各个类别的概率
        Class_each={}
        for y in probability_y:
            Class_each[y]=probability_y[y]
            for x in features:
                print(Class_each[y])
                print(probability[str(x)+'|'+str(y)])
                Class_each[y] =Class_each[y]*probability[str(x)+'|'+str(y)]

        features_label =max(Class_each,key=Class_each.get)
        return features_label
    
    
    
if __name__ == '__main__':
    nb=NaiveBayes()
    #训练数据
    trainData,labels = nb.getTrainSet()
    #x1,x2
    features = [2,'S']
    #该特征应属于哪一类
    result = nb.classify(trainData,labels,features)
    print(features)
    print('属于')
    print(result)








            



        
#%%
#专用官方
# http://mpld3.github.io/quickstart.html

import matplotlib.pyplot as plt
import numpy as np
import mpld3


mpld3.enable_notebook()

fig, ax = plt.subplots(subplot_kw=dict(axisbg='#EEEEEE'))
ax.grid(color='white', linestyle='solid')
N = 50
scatter = ax.scatter(np.random.normal(size=N),
                     np.random.normal(size=N),
                     c=np.random.random(size=N),
                     s = 1000 * np.random.random(size=N),
                     alpha=0.3,
                     cmap=plt.cm.jet)
ax.set_title("D3 Scatter Plot", size=18);

#%%
#http://bokeh.pydata.org/en/latest/docs/gallery.html
#专门官网
from bokeh.io import push_notebook, show, output_notebook
from bokeh.layouts import row, gridplot
from bokeh.plotting import figure, show, output_file
output_notebook()

import numpy as np

x = np.linspace(0, 4*np.pi, 100)
y = np.sin(x)
TOOLS = "pan,wheel_zoom,box_zoom,reset,save,box_select"

p1 = figure(title="Legend Example", tools=TOOLS)
p1.circle(x,   y, legend="sin(x)")
p1.circle(x, 2*y, legend="2*sin(x)", color="orange")
p1.circle(x, 3*y, legend="3*sin(x)", color="green")
show(p1)


#%%
from IPython.display import Latex
Latex('''The mass-energy equivalence is described by the famous equation

$$E=mc^2$$

discovered in 1905 by Albert Einstein.
In natural units ($c$ = 1), the formula expresses the identity

\\begin{equation}
E=m
\\end{equation}''')


#%%
#Inline images
from IPython.display import Image
Image('http://jakevdp.github.com/figures/xkcd_version.png')


#%%
#IFrame
from IPython.core.display import HTML
HTML("<iframe src='http://www.ncdc.noaa.gov/oa/satellite/satelliteseye/cyclones/pfctstorm91/pfctstorm.html' width='750' height='600'></iframe>")


#%%
#朴素贝叶斯算法   贝叶斯估计， λ=1  K=2， S=3； λ=1 拉普拉斯平滑
import pandas as pd
import numpy as np

class NavieBayesB(object):
    def __init__(self):
        self.A = 1    # 即λ=1
        self.K = 2
        self.S = 3

    def getTrainSet(self):
        trainSet = pd.read_csv('C://pythonwork//practice_data//naivebayes_data.csv')
        trainSetNP = np.array(trainSet)     #由dataframe类型转换为数组类型
        trainData = trainSetNP[:,0:trainSetNP.shape[1]-1]     #训练数据x1,x2
        labels = trainSetNP[:,trainSetNP.shape[1]-1]          #训练数据所对应的所属类型Y
        return trainData, labels

    def classify(self, trainData, labels, features):
        labels = list(labels)    #转换为list类型
        #求先验概率
        P_y = {}
        for label in labels:
            P_y[label] = (labels.count(label) + self.A) / float(len(labels) + self.K*self.A)

        #求条件概率
        P = {}
        for y in P_y.keys():
            y_index = [i for i, label in enumerate(labels) if label == y]   # y在labels中的所有下标
            y_count = labels.count(y)     # y在labels中出现的次数
            for j in range(len(features)):
                pkey = str(features[j]) + '|' + str(y)
                x_index = [i for i, x in enumerate(trainData[:,j]) if x == features[j]]   # x在trainData[:,j]中的所有下标
                xy_count = len(set(x_index) & set(y_index))   #x y同时出现的次数
                P[pkey] = (xy_count + self.A) / float(y_count + self.S*self.A)   #条件概率

        #features所属类
        F = {}
        for y in P_y.keys():
            F[y] = P_y[y]
            for x in features:
                F[y] = F[y] * P[str(x)+'|'+str(y)]

        features_y = max(F, key=F.get)   #概率最大值对应的类别
        return features_y


if __name__ == '__main__':
    nb = NavieBayesB()
    # 训练数据
    trainData, labels = nb.getTrainSet()
    # x1,x2
    features = [2,'S']
    # 该特征应属于哪一类
    result = nb.classify(trainData, labels, features)
    print features,'属于',result
