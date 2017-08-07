# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as ny
import numpy
import operator
from os import listdir
import matplotlib
import matplotlib.pyplot as plt
def classKnn(vectorX,dataSet,labels,k):#分别表示测试向量，已经格式化好的数据集，格式化好的标签，以及可以调试的参数k
    dataSetSize=dataSet.shape[0]#提取数据集的行数
    diffMat=ny.tile(vectorX,(dataSetSize,1))-dataSet#对于测试向量，进行行扩展
    sqDiffMat=diffMat**2
    sqDistances=sqDiffMat.sum(axis=1)
    distances=sqDistances**0.5#上面是计算距离，各个坐标相减，然后平方，最终开根号得出每个点的距离
    sortedDistIndices=distances.argsort()#对于距离进行排队，返回从小到大排列的序列
    classCount={}
    for i in range(k):
    	voteIlabel=labels[sortedDistIndices[i]]#从序列中取出对应数值，然后赋值
    	classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
   # print("sortdClass")           
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    #classCount字典分解为元祖列表，对元素的第二个元素进行排序，为逆序
   # print("ok")
    return sortedClassCount[0][0]

#print("ok")
group=ny.array([[1,1.1],
             [1,1],
             [0,0],
             [0,0.1]])
labels=['A','A','B','B']
def fileOperator(filename):#注意的问题，我们传入的应该是数据，而不是字符串
    fread=open(filename)#打开文件
    arrayEachLines=fread.readlines()#读取行数
    numberEachLines=len(arrayEachLines)#获得行数
    getMatrix=numpy.zeros((numberEachLines,3))#创建零矩阵
    classLabelVector=[]
    index=0
    for line in arrayEachLines:#对于每一行进行数据整理
    	line=line.strip()#删除开头处和结尾处多余的空格，以及回车符号，形成一行
    	listFromLine=line.split('\t')#加入制表格分开
    	getMatrix[index,:]=listFromLine[0:3]#前三个数据分为一行
    	classLabelVector.append(int(listFromLine[-1]))#-1表示最后一列的元素
    	index += 1
    return getMatrix,classLabelVector
#classKnn([0,0],group,labels,3)



def normalization(dataSet):#归一化特征
	minValues=dataSet.min(0)#参数0表示取每一列的最小值
	maxValues=dataSet.max(0)
	range=maxValues- minValues
	#m=dataSet.shape
	newMatrix=numpy.zeros(dataSet.shape)#直接把大小投入进去更为方便不出bug
	m=dataSet.shape[0]
	newMatrix=dataSet- numpy.tile(minValues,(m,1))#矩阵减去最小值
	newMatrix=newMatrix/numpy.tile(range,(m,1))#矩阵所有元素除以最小值与最大值之差
	return newMatrix,range,minValues

def datingClassTest():
	holdRadio=0.5
	datingDataMatrix,datingLabels=fileOperator("datingTestSet2.txt")#取出数据集
	normalizationMatrix,ranges,minValues=normalization(datingDataMatrix)#数据集归一化处理
	m=normalizationMatrix.shape[0]#获得矩阵行数
	numTestVecs=int(m*holdRadio)#测试数据为行数的一半
	errorCount=0#表示最终结果错误率
	for i in range(numTestVecs): #在前行数的一半范围
		classifierResult = classKnn(normalizationMatrix[i,:],normalizationMatrix[numTestVecs:m,:]
			,datingLabels[numTestVecs:m],3)
		print("the classifier came back with: %d, the real answer is: %d"
		 % (classifierResult,datingLabels[i]))
		if (classifierResult != datingLabels[i]):
			errorCount+=1
	print("the total error rate is %f" % (errorCount/float(numTestVecs)))
	print(errorCount)
"""
	resultList=['not','small doses','large doses']
	percent =float(input("percent"))
	mile =float(input("mile"))
	iceCream =float(input("iceCream"))
	#inArr=numpy.array([mile,percent,iceCream])
	#classifierResult=classKnn((inArr[0] - minValues)/range,normalizationMatrix,datingLabels,3)
   inArr=numpy.array([(mile- minValues)/range,percent,iceCream])
	classifierResult=classKnn((inArr,normalizationMatrix,datingLabels,3)
	print("you like the person",resultList[classifierResult-1])
"""
"""
def classifyPerson():
	resultList=['not','small doses','large doses']
	percent =float(input("percent"))
	mile =float(input("mile"))
	iceCream =float(input("iceCream"))
	inArr=array([mile,percent,iceCream])
	classifierResult=classKnn((inArr-minValues)/range,normalizationMatrix,datingLabels,3)
	print("you like the person",resultList[classifierResult-1])
    """
def main():
   # return classKnn([0,0],group,labels,3)
   
   datingDataMatrix,datingDataVector=fileOperator("datingTestSet2.txt")
   fig=plt.figure()
   ax=fig.add_subplot(111)
   ax.scatter(datingDataMatrix[:,1],datingDataMatrix[:,2],
   	          15*numpy.array(datingDataVector),15*numpy.array(datingDataVector))
   datingClassTest()
   #ax.scatter(datingDataMatrix[:,1],datingDataMatrix[:,2])
  # plt.show()
   return 0

print(main())

