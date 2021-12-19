import matplotlib.pyplot as plt
import  numpy as np
import pandas as pa

fig, axes = plt.subplots(nrows=2, ncols=3)
g=0
for i in range(2):
    for j in range(3):
        neuron = np.loadtxt("Neuron%d.txt" % (g+1), dtype=np.double)
        im = axes[i, j].imshow(neuron, cmap=plt.cm.viridis, vmin=-2, vmax=3)
        axes[i, j].title.set_text("Neuron %d" % (g+1))
        axes[i, j].invert_yaxis()
        g = g+1
plt.colorbar(im, ax=axes.ravel().tolist())


plt.show()

