# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 17:29:55 2017

@author: Magneto_Wang
"""
import PlotTrees
from math import log
import operator
def calculedShannonEnt(dataSet):#计算香农熵
 	numberEntries=len(dataSet)#取得数据集合大小
 	labelCounts={}#获取对应特征的列表
 	i=0
 	for featureVector in dataSet:
# 		print(dataSet)
# 		print("feature = %d"%i)
# 		i += 1
# 		print(featureVector)
 		currentLabel=featureVector[-1]#从新获取的一行的最后一个元素选取
 		if currentLabel not in labelCounts.keys():
 			#如果获取的元素不在建立的特征列表，那么就新建立一个
 			labelCounts[currentLabel]=0
 		labelCounts[currentLabel]+=1#如果存在，那么增加一个
 	shannonEntropy =0#设立香农熵的值，信息的期望值
   
 	for key in labelCounts:
 		#print(key)
 		prob=float(labelCounts[key]/numberEntries)#求出每一个特征在列表所占的概率
 		shannonEntropy-= prob*log(prob,2)#对概率按照公式进行计算。然后就和，就是熵
 	return shannonEntropy#返回熵值

def splitDataSet(dataSet,axis,value):
	#将要划分的数据集，和指定要划分的特征，以及返回的特征
		normalizeDataSet=[]#初始化数据集
		for featureVector in dataSet:#抽取一行
			if featureVector[axis] == value:#在指定行找出特征，并进行比较
			#如果相等，那么抽出除这个特征以外的列表
				reducedFeatureVector =featureVector[:axis]
				reducedFeatureVector.extend(featureVector[axis+1:]) 
				normalizeDataSet.append(reducedFeatureVector)
#		print(normalizeDataSet)
#		print("wrongbegin")
#		print("wrongend")

		return normalizeDataSet#返回错误的集合了。这里出了bug

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    
    return dataSet, labels


def chooseBestFeatureToSplit(dataSet):
	numberFeature =len(dataSet[0])-1#获取特征数量
	baseEntropy =calculedShannonEnt(dataSet)#计算熵
	bestInfomationGain =0#设置信息增益
	bestFeature =-1
	for i in range(numberFeature):
		featList = [example[i] for example in dataSet]#抽取列表的基本语法
		uniqueValues = set(featList)#设置集合，删除重复特征
		newEntropy = 0
		for value in uniqueValues:
			subDataSet = splitDataSet(dataSet,i,value)
			#将要划分的数据集，和指定要划分的特征，以及返回的特征
			prob = len(subDataSet)/float(len(dataSet))
#			print("subDataSet")
#			print(subDataSet)
#			print("subDataSet")
#			print("wrongBegin")
#			print("wrongMedia")
#			print("wrongEnd")

			newEntropy += prob*calculedShannonEnt(subDataSet)
		infoGain = baseEntropy - newEntropy#计算信息增益
		if(infoGain>bestInfomationGain):#选择信息增益最大的作为最好的特征
			bestInfomationGain=infoGain
			bestFeature=i
	return bestFeature

def majorityCnt(classList):#选择特征量最多的一类，并且排序
	classCount={}
	for vote in classList:
		if vote not in classCount.keys(): classCount[vote]=0
		classCount[vote] += 1
		sortedClassCount = sorted(classCount.item(),key=operator.itemgetter(1),
			reverse=True)
	return sortedClassCount[0][0]

def creatTree(dataSet,labels):#创建树，数据集和标签列表
	classList = [example[-1] for example in dataSet]#提取每一行的最后一个元素，也就是标签
	if classList.count(classList[0]) == len(classList):
	#count() 方法用于统计某个元素在列表中出现的次数
	#如果相等，就说明进入到同一划分的类别
		return classList[0]
	if len(dataSet[0]) == 1:#说明进入叶子节点，不可继续划分
		return majorityCnt(classList)#只好返回特征值量最多的
	bestFeature = chooseBestFeatureToSplit(dataSet)#在仍然可以划分数据集中找最好的
	bestFeatureLabel = labels[bestFeature]#找出对应标签
	myTree = {bestFeatureLabel:{}}#创建字典，字典里面的值仍然是字典。形成嵌套
	del(labels[bestFeature])#删除这个特征标签
	featureValues = [example[bestFeature] for example in dataSet]#对应标签的值
	uniqueValues = set(featureValues)
	for value in uniqueValues:
		subLabels = labels[:]
		print(subLabels)
		print("   ")
#		print(dataSet)
		myTree[bestFeatureLabel][value] = creatTree(splitDataSet(dataSet,bestFeature,value),subLabels)

	return myTree

def classify(inputTree,featureLabels,testVector):
	firstStr = inputTree.keys()[0]#取出关键词的第一个值
	secondDict = inputTree[firstStr] #关键词对应的元素仍然是字典
	featureIndex = featureLabels.index(firstStr)#索引对应的元素
	key = testVector[featureIndex]#在对应元素继续取值
	valueOfFeature = secondDict[key]#取出的值，继续索引
	if isinstance(valueOfFeature,dict):#判定新的元素是否为字典。也就是说，是否到叶子结点
		classLabel = classify(valueOfFeature,featureLabels,testVector)#否则继续递归
	else: classLabel = valueOfFeature#叶子结点到，准备返回
	return classLabel

def storeTree(inputTree,filename):#存储构造好的树的结构
	import pickle
	fileWrite = open(filename,'w')
	pickle.dump(inputTree,fileWrite)
	fileWrite.close()

def grabTree(filename):#读取树的结构
	import pickle
	fileRead = open(filename)
	return pickle.load(fileRead)

def main():
	fr=open('lenses.txt')
	lenses = [inst.strip().split('\t') for inst in fr.readlines()]

	lensesLabel = ['age','prescript','astigmatic','tearRate']
	print(lensesLabel)
	lensesTree = creatTree(lenses,lensesLabel)
	print(lensesTree)
	PlotTrees.createPlot(lensesTree)    
    #dataSet,labels=createDataSet()
    #answer=calculedShannonEnt(dataSet)
#    print(dataSet)
#    print("wrongBegin")
#    print("wrongBegin")
#    print("wrongBegin")
    #print(chooseBestFeatureToSplit(dataSet))
   # print(answer)
    #return 0
    
main()