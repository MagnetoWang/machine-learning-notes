# -*- coding: utf-8 -*-
"""
Created on Thu May 24 12:31:18 2018

@author: Magneto_Wang
"""

import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np
from PIL import Image
from sklearn.utils.extmath import stable_cumsum


def pca(data,n_components):
    mean=np.mean(data, axis=0)
    #print("mean.shape : "+str(mean.shape))
    
    
    X=data


    n_samples, n_features = X.shape
    X=X.astype(float) 
    X -= mean
    U, S, V = np.linalg.svd(X, full_matrices=False)
    components = V

    explained_variance_ = (S ** 2) / (n_samples - 1)
    total_var = explained_variance_.sum()
    explained_variance_ratio_ = explained_variance_ / total_var
    #singular_values_ = S.copy()

    ratio_cumsum = stable_cumsum(explained_variance_ratio_)
    n_components = np.searchsorted(ratio_cumsum, n_components) + 1
    #noise_variance = explained_variance_[n_components:].mean()
    print("U")
    print(U.shape)
    print("S")
    print(S.shape)
    print("V")
    print(V.shape)

    components=components[0:n_components+1,:]
    #components.shape
    X_transformed = np.dot(X, components.T)
    print(components.shape)
    print("X_transformed.shape")
    print(X_transformed.shape)
    #plt.imshow(X_transformed,cmap='Greys_r')
    #plt.show()
    
    newimage = X_transformed[0,:]
    newimage=newimage.reshape(1,83)
    
    print(newimage.shape)
    print("components")
    print(components.shape)
    reconstrutX=np.dot(X_transformed, components) + mean
    reimage=np.dot(newimage, components) + mean
    print("reconstrutX.shape")
    print(reconstrutX.shape)
    reimage=reimage.reshape(384,256)

    plt.imshow(reimage,cmap='Greys_r')
    plt.show()
    return 

    

image=mpimg.imread('0.jpg')
n_components =0.2
plt.imshow(image,cmap='Greys_r')
plt.show()
'''    
image=mpimg.imread('0.jpg')
n_components =0.95
plt.imshow(image,cmap='Greys_r')
plt.show()
pca(image,n_components)
'''
#print(image.shape)
def numberofImage(image):
    data=np.array(image.reshape(1,384*256))
    data=data
    #print(data.shape)
    for i in range(100):
        filename='灰度图片/'+str(i)+'.jpg'
        #print(filename)
        image=mpimg.imread(filename)
        #print(image.shape)
        #print("data : ")
        #print(data.shape)
        data=np.vstack((data,image.reshape(1,384*256)))
        #data=np.vstack((data,image))
        #print(data.shape)

    print("last")
    print(data.shape)
    pca(data,0.95)
    #np.savetxt("data.txt",data)
    
    
numberofImage(image)
# 1:numpy.savetxt(fname,X):第一个参数为文件名，第二个参数为需要存的数组（一维或者二维）。

# 2.numpy.loadtxt(fname)：将数据读出为array类型。