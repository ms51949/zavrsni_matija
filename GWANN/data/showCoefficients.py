import matplotlib.pyplot as plt
import  numpy as np
import pandas as pa

fig, axes = plt.subplots(nrows=2, ncols=2)
min = 0.5
max = 4
beta0 = np.loadtxt("beta0.txt", dtype=np.double)
im = axes[0,0].imshow(beta0, cmap=plt.cm.viridis, vmin=min, vmax=max)

beta1 = np.loadtxt("beta1.txt", dtype=np.double)
im = axes[0,1].imshow(beta1, cmap=plt.cm.viridis, vmin=min, vmax=max)

beta12 = np.loadtxt("beta12.txt", dtype=np.double)
im = axes[1,0].imshow(beta12, cmap=plt.cm.viridis, vmin=min, vmax=max)

beta22 = np.loadtxt("beta22.txt", dtype=np.double)
im = axes[1,1].imshow(beta22, cmap=plt.cm.viridis, vmin=min, vmax=max)

axes[0,0].invert_yaxis()
axes[0,1].invert_yaxis()
axes[1,0].invert_yaxis() 
axes[1,1].invert_yaxis() 

plt.colorbar(im, ax=axes.ravel().tolist())
plt.show()

