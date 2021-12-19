import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1,3)
matrica = np.loadtxt("toy2Coef.txt", dtype=np.double)
beta0 = np.reshape(matrica[:, 0], (25, 25))
beta1 = np.reshape(matrica[:, 1], (25, 25))
beta22 = np.reshape(matrica[:, 2], (25, 25))
im = axes[0].imshow(beta0, cmap=plt.cm.viridis, vmin=0.5, vmax=3.5)
axes[0].title.set_text("β0")
im = axes[1].imshow(beta1, cmap=plt.cm.viridis, vmin=0.5, vmax=3.5)
axes[1].title.set_text("β1")
im = axes[2].imshow(beta22, cmap=plt.cm.viridis, vmin=0.5, vmax=3.5)
axes[2].title.set_text("β22")

plt.colorbar(im, ax=axes.ravel().tolist())

axes[0].invert_yaxis()
axes[1].invert_yaxis()
axes[2].invert_yaxis()

plt.show()