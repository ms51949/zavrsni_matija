import numpy as np
import math


class LossFun:
    
    def __init__(self, locations, bandwitdh):
        self.bandwitdh = bandwitdh
        self.size = locations.shape[0]
        self.matrica = self.mat(locations)

    def mse(self,y_true, y_pred, rowNum):
        return np.sum(self.getMatrica(rowNum) * np.power(y_true-y_pred, 2))/2


    def mse_prime(self,y_true, y_pred, rowNum):
        return (self.getMatrica(rowNum)*(y_pred - y_true))

    def matricaVij(self, l, locations):
        matrica = np.zeros((self.size, 1))
        m = 0
        for location in locations:
            distance = np.linalg.norm(l-location)
            matrica[m] = math.exp(-0.5*((distance/self.bandwitdh)**2))
            m = m+1
        return matrica

    def mat(self, locations):
        matrica = np.zeros((self.size, self.size, 1))
        i=0
        for location in locations:
            matrica[i] = self.matricaVij(location, locations)
            i =  i + 1
        return matrica

    def getMatrica(self, rowNum):
        rowNum = int(rowNum)
        return np.reshape(self.matrica[rowNum], (625,1))