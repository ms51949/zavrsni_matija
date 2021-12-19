import pandas as pa
import numpy as np
import matplotlib.pyplot as plt
import math


data = pa.read_csv("toy4.csv")
X_train = np.array(data[['x1', 'x2']])
Y_train = np.array(data[['y']])
LOCATION_train = np.array(data[['lon', 'lat']])


def Wi(l, locations, bandwitdh):
    matrica = np.zeros((locations.shape[0], locations.shape[0]))
    m = 0
    for location in locations:
        distance = np.linalg.norm(l-location)
        matrica[m][m] = math.exp(-0.5*((distance/bandwitdh)**2))
        m = m+1
    return matrica


def fit(X, y, locations, location, bandwitdh):
    w = Wi(location, locations, bandwitdh)
    pom1 = np.linalg.inv(np.linalg.multi_dot([X.T, w, X]))
    pom2 = np.linalg.multi_dot([X.T, w, y])
    beta = np.dot(pom1, pom2)
    return beta

def fitData(X, y, locations, bandwitdh):
    m, n=X.shape
    matricaPrediction = np.zeros((m,n+1, 1))
    ones = np.ones((m, 1))
    X_ = np.hstack((ones, X))
    m=0
    for location in locations:
        matricaPrediction[m] = fit(X_, y, locations,location, bandwitdh)
        m = m + 1
    matricaPrediction = np.reshape(matricaPrediction, (m,n+1))
    return matricaPrediction


matrica = fitData(X_train, Y_train, LOCATION_train, 2.0)
np.savetxt("toy4Coef.txt",matrica)
