# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 10:37:11 2017

@author: Magneto_Wang
"""

from numpy import *






def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
                 ]
    classVector = [0,1,0,1,0,1]#1代表侮辱，0代表正常
    return postingList,classVector
    

def createVocabList(dataSet):
    
    vocabSet = set([])
    for document in dataSet:
        vocabSet=vocabSet | set(document)
    return list(vocabSet)
	
    
	
        
    


def setOfWord2Vector(vocabList,inputSet):
    returnVector = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVector[vocabList.index(word)] = 1
        else:print('the word %s is not in my vocabulary'% word)
    return returnVector


def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        
        if (trainCategory[i] == 1):
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
    else:
    	p0Num += trainMatrix[i]
    	p0Denom += sum(trainMatrix[i])
    p1Vector = log(p1Num/p1Denom)
    p0Vector = log(p0Num/p0Denom)
    return p0Vector,p1Vector,pAbusive


def classifyNB(vector2Classify,p0Vector,p1Vector,pClass1):
    p1 = sum(vector2Classify*p1Vector)+log(pClass1)
    p0 = sum(vector2Classify *p0Vector)+log(1.0- pClass1)
    if (p1>p0):
        
        return 1
    else:
        return 0
    	

def  bagOFwords2VectorMN(vocabList,inputSet):
    returnVector  = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVector[vocabList.index(word)] += 1
    return returnVector     


def testingNB():
    listOfPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOfPosts)
    trainMatrix = []
    for postinDoc in listOfPosts:
        trainMatrix.append(setOfWord2Vector(myVocabList,postinDoc)) 
    p0Vector,p1Vector,pAbusive = trainNB0(array(trainMatrix),array(listClasses))
    testEntry = ['love','my','dalmation']
    thisDoc = array(setOfWord2Vector(myVocabList,testEntry))
    print(testEntry)
    print('classified as :')
    print(classifyNB(thisDoc,p0Vector,p1Vector,pAbusive))
    testEntry = ['stupid','garbage']
    thisDoc = array(setOfWord2Vector(myVocabList,testEntry))
    print(testEntry)
    print('classified as :')
    print(classifyNB(thisDoc,p0Vector,p1Vector,pAbusive))

def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*',bigString)
    return [tok.lower() for tok in listOfTokens if (len(tok)>2)]

def spamTest():
	
    
	
testingNB()

	
	

                
                
                
        	
        	
        
        
	
        


		
       
			
        
        
        
    
    
    
    
    
    
