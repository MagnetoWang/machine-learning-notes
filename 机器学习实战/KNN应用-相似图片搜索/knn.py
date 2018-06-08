from collections import Counter
import numpy as np
from math import *

class KNNClassifier:

    def __init__(self,k):

        self.k = k
        self._X_train = None
        self._y_train = None
    
    def fit(self, X_train, y_train):
        self._X_train = X_train
        self._y_train = y_train
        return self
    
    def predict(self, X_predict):
        
        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)

    def _predict(self, x):
        
        distances = [sqrt(np.sum((x_train - x) ** 2)) for x_train in self._X_train]
        nearest = np.argsort(distances)

        topK_y = [self._y_train[i] for i in nearest[:self.k]]
        votes = Counter(topK_y)

        return votes.most_common(1)[0][0]
    
    def __repr__(self):
        return "KNN(K=%d)" % self.k

