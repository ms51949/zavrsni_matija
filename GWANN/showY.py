import matplotlib.pyplot as plt
import  numpy as np
import pandas as pa

fig, axes = plt.subplots(1,4)
for i in range(4):
    data = pa.read_csv("toy%d.csv" % (i+1))
    Y_train = np.array(data[['y']])
    LOCATION_train = np.array(data[['lon', 'lat']])


    matrica = np.reshape(Y_train, (25,25))
    im = axes[i].imshow(matrica, cmap=plt.cm.viridis)
#plt.colorbar(im, ax=axes.ravel().tolist())
plt.show()