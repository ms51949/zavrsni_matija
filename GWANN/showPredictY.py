import matplotlib.pyplot as plt
import  numpy as np
import pandas as pa

fig, axes = plt.subplots()
data = np.loadtxt("PredictY.txt", dtype=np.double)
matrica = np.zeros((625,1))
for i in range(625):
    matrica[i][0] = data[i][i]
matricaPred = np.reshape(matrica, (25,25))

data = pa.read_csv("toy4.csv")
Y_train = np.array(data[['y']])
matricaY = np.reshape(Y_train, (25,25))

axes.imshow(abs(matricaPred-matricaY), cmap=plt.cm.viridis)
axes.invert_yaxis
plt.show()
