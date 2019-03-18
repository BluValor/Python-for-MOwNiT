import numpy as np
import random
import matplotlib.pyplot as mp
import time

m_size = 10

A = np.zeros([10, 10])
B = np.full((m_size, 1), 1.0)

for m in range(m_size):
    for n in range(m_size):
        A[m][n] = random.random() * 100.0


def zero_column(mA, mB, n, size, m):
    for i in range(size):
        if i is not m:
            for j in range(size):
                mA[j][i] = mA[j][i] + (-A[j][m] / A[n][m] * A[n][i])


def gj_column(mA, mB, n, size):
    c_max = 0
    for i in range(size):
        if A[n][i] > A[n][c_max]:
            c_max = i


print(A)
print(B)


