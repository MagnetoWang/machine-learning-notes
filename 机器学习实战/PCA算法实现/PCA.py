# -*- coding: utf-8 -*-
"""
Created on Thu May 24 11:15:11 2018

@author: Magneto_Wang
"""

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np
from PIL import Image

image=mpimg.imread('0.jpg')
image.shape
image
plt.imshow(image)
plt.show()
# Make an instance of the Model
pca = PCA(.95)
pca.fit(image)
print("fit image")
print(image.shape)
image = pca.transform(image)
print("transform image")
print(image.shape)


plt.imshow(image)
plt.show()


approximation = pca.inverse_transform(image)
print("inverse_transform image")
print(approximation.shape)
plt.imshow(approximation)
#plt.axis('off')
plt.show()