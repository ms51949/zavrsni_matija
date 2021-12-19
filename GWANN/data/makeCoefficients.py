from coefficients import beta0, beta1, beta12, beta22
import numpy as np

matrix0 = np.zeros((25, 25))
matrix1 = np.zeros((25, 25))
matrix12 = np.zeros((25, 25))
matrix22 = np.zeros((25, 25))

for i in range(25):
    for j in range(25):
        matrix0[i][j] = beta0()
        matrix1[i][j] = beta1(i, j)
        matrix12[i][j] = beta12(i, j)
        matrix22[i][j] = beta22(i, j)

np.savetxt("GWANN//data//beta6dd9.txt", matrix0)
np.savetxt("GWANN//data//beta1.txt", matrix1)
np.savetxt("GWANN//data//beta12.txt", matrix12)
np.savetxt("GWANN//data//beta22.txt", matrix22)