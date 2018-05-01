#%%
from IPython.display import Latex
#https://matplotlib.org/users/usetex.html
#https://matplotlib.org/tutorials/index.html#introductory
# http://scikit-learn.org/stable/auto_examples/neighbors/plot_kde_1d.html
#for latex
Latex('''
Parzen窗估计属于非参数估计。所谓非参数估计是指，已知样本所属的类别，

但未知总体概率密度函数的形式，要求我们直接推断概率密度函数本身。

非参数估计的方法主要有：直方图法、核方法。Parzen窗估计属于核方法的一种


非参数估计的方法主要有：直方图法、核方法。Parzen窗估计属于核方法的一种
$$$$
Kernel density estimation
 is a non-parametric way to estimate the probability density function of a random variable. Kernel density estimation is a fundamental data smoothing problem where inferences about the population are made, based on a finite data sample. In some fields such as signal processing and econometrics it is also termed the Parzen–Rosenblatt window method, after Emanuel Parzen and Murray Rosenblatt, who are usually credited with independently creating it in its current form
\\begin{equation}

\\end{equation}

\\begin{equation}

\\end{equation}
对于parzen知识点，更好的说法应该是核密度估计。
$$核心是这个公式$$

$${\displaystyle {\hat {f}}_{h}(x)={\\frac {1}{n}}\sum _{i=1}^{n}K_{h}(x-x_{i})={\\frac {1}{nh}}\sum _{i=1}^{n}K{\Big (}{\\frac {x-x_{i}}{h}}{\Big )},}$$

where K is the kernel — a non-negative function that integrates to one — and h > 0 is a smoothing parameter called the bandwidth.

A kernel with subscript h is called the scaled kernel and defined as $Kh(x) = 1/h K(x/h)$.

''')







#%%
from scipy import signal
from scipy.fftpack import fft, fftshift
import matplotlib.pyplot as plt
#Number of points in the output window. If zero or less, an empty array is
#returned.
window = signal.parzen(51)
plt.plot(window)
plt.title("Parzen window")
plt.ylabel("Amplitude")
plt.xlabel("Sample")
plt.figure()


window = signal.parzen(100)
plt.plot(window)
plt.title("Parzen window")
plt.ylabel("Amplitude")
plt.xlabel("Sample")
plt.figure()
# A = fft(window, 2048) / (len(window)/2.0)
# freq = np.linspace(-0.5, 0.5, len(A))
# response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
# plt.plot(freq, response)
# plt.axis([-0.5, 0.5, -120, 0])
# plt.title("Frequency response of the Parzen window")
# plt.ylabel("Normalized magnitude [dB]")
# plt.xlabel("Normalized frequency [cycles per sample]")



#%%
# http://python.jobbole.com/81321/
#生成正态分布

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from numpy.linalg import cholesky  
mu= 0
sigma =1
x = np.arange(0,1,0.00001)
y =norm.pdf(x,0.5,1)

plt.plot(x,y)
plt.title("[0,1]之间的正态分布")
plt.ylabel("y")
plt.xlabel("x")
plt.show()

sampleNo = 1000;
mu = 0
sigma = 1
np.random.seed(0)
s = np.random.normal(mu, sigma, sampleNo )
plt.hist(s, 30, normed=True)
# plt.plot(s)
# plt.show()
# window = signal.parzen(51)
# plt.plot(window)
# plt.title("Parzen window")
# plt.ylabel("Amplitude")
# plt.xlabel("Sample")
# plt.figure()


# sampleNo = 10000;
# mu = 3  
# sigma = 0.1  
# np.random.seed(0)  
# s = np.random.normal(mu, sigma, sampleNo )  
# plt.subplot(141)  
# plt.hist(s, 30, normed=True)  

# np.random.seed(0)  
# s = sigma * np.random.randn(sampleNo ) + mu  
# plt.subplot(142)  
# plt.hist(s, 30, normed=True)  
  
# np.random.seed(0)  
# s = sigma * np.random.standard_normal(sampleNo ) + mu  
# plt.subplot(143)  
# plt.hist(s, 30, normed=True)  
  
# # 二维正态分布  
# mu = np.array([[1, 5]])  
# Sigma = np.array([[1, 0.5], [1.5, 3]])  
# R = cholesky(Sigma)  
# s = np.dot(np.random.randn(sampleNo, 2), R) + mu  
# plt.subplot(144)  
# # 注意绘制的是散点图，而不是直方图  
# plt.plot(s[:,0],s[:,1],'+')  
# plt.show()  





