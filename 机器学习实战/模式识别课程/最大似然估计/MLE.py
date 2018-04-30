
#%%
from IPython.display import Latex
#https://zh.wikipedia.org/wiki/%E6%9C%80%E5%A4%A7%E4%BC%BC%E7%84%B6%E4%BC%B0%E8%AE%A1
#https://blog.csdn.net/zengxiantao1994/article/details/72787849
Latex('''
最大似然估计（英语：maximum likelihood estimation，缩写为MLE），也称最大概似估计，是用来估计一个概率模型的参数的一种方法。


如概率分布、概率密度函数、随机变量、数学期望等。同时，还要求读者熟悉连续实函数的基本技巧，比如使用微分来求一个函数的极值（即极大值或极小值）。



最大似然估计的原理
给定一个概率分布  $ \\boxed{D}$，已知其概率密度函数（连续分布）或概率质量函数（离散分布）为 $f_D$ ，
以及一个分布参数 ${\\theta }$  ，
我们可以从这个分布中抽出一个具有 ${ n}$ 个值的采样 
$${ X_{1},X_{2},\\ldots ,X_{n}}$$，利用 $f_D$ 计算出其似然函数：

$${ {\\mbox{lik}}(\\theta \\mid x_{1},\\dots ,x_{n})=f_{\\theta }(x_{1},\\dots ,x_{n}).}$$

若 ${ D}$ 是离散分布，  $f_{\\theta }$即是在参数为 { \\theta } & 时观测到这一采样的概率。


若其是连续分布， {\displaystyle f_{\theta }} {\displaystyle f_{\theta }}则为 {\displaystyle X_{1},X_{2},\ldots ,X_{n}} X_1, X_2,\ldots, X_n联合分布的概率密度函数在观测值处的取值。

一旦我们获得 {\displaystyle X_{1},X_{2},\ldots ,X_{n}} X_1, X_2,\ldots, X_n，我们就能求得一个关于 {\displaystyle \theta } \theta 的估计。

最大似然估计会寻找关于 {\displaystyle \theta } \theta 的最可能的值（即，在所有可能的 {\displaystyle \theta } \theta 取值中，寻找一个值使这个采样的“可能性”最大化）。从数学上来说，我们可以在 {\displaystyle \theta } \theta 的所有可能取值中寻找一个值使得似然函数取到最大值。这个使可能性最大的 {\displaystyle {\widehat {\theta }}} \widehat{\theta}值即称为 {\displaystyle \theta } \theta 的最大似然估计。由定义，最大似然估计是样本的函数。




$$核心是实现这个正态分布的最大相似函数$$
$${\\theta}=(\widehat{\mu},\widehat{\\sigma}^2) = (\\bar{x},\sum_{i=1}^n(x_i-\\bar{x})^2/n) $$

''')



#%%
import os
import pandas as pd
import numpy as np
import math
# __file__="MLE.py"
# print(os.path.realpath("MLE.py")) #获取当前文件路径
# print(os.path.dirname(os.path.realpath(__file__)))  # 从当前文件路径中获取目录
# print(os.path.basename(os.path.realpath(__file__))) #获取文件名
    
    def getDataSet(self):
        dataSet = pd.read_csv("MLE_data.csv")
        print("dataSet")
        #print(dataSet)
        print("dataSet end")
        dataSetArray=np.array(dataSet)
        return dataSetArray



    def getUandC(self,features):
    #mu为期望向量，sigma为协方差矩阵，n为规模
        #     m_hat：样本由极大似然估计得出的正态分布参数，均值  
        # s_hat：样本由极大似然估计得出的正态分布参数，方差  
        mu=1
        sigma=1
        print("features")
        print(features)
        feature_number = [float(x) for x in features if x==x]
        mu=sum(feature_number)/len(feature_number)
        sums=0
        for i in feature_number:
            sums=sums+pow((i-mu),2)

        sigma=sums/len(features)
        
            



        return mu,sigma

    def getSample(self,Sample):
        print("Sample")
        print(Sample)
        for i in range(len(Sample[0])): 
            features = Sample[0:,i]
            mu,sigma=self.getUandC(features)
            print("  mu = "+str(mu)+"  = sigma "+str(sigma))




        return mu,sigma
        

# 类的方法的调用

# 与普通的函数调用类似

# 1.类的内部调用：self.<方法名>(参数列表)。

# 2.在类的外部调用：<实例名>.<方法名>(参数列表)。

    def getAllSample(self,dataSetArray):
        
        Sample = dataSetArray[0:10:1,:]
        print("NO.1")
        # print(Sample)
        self.getSample(Sample)
        Sample = dataSetArray[11:21,:]
        print("NO.2")
        # print(Sample)
        self.getSample(Sample)
        Sample = dataSetArray[22:32,:]
        print("NO.3")
        # print(Sample)
        self.getSample(Sample)


        return 0
        
        
        



if __name__ == '__main__':
    ml=MLE()
    datatrain=ml.getDataSet()
    print(ml.getAllSample(datatrain))

        
        
