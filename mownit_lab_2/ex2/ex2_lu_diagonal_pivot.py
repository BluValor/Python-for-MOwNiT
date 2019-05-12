import numpy as np
import scipy as sc
import scipy.linalg as lg

np.set_printoptions(precision=3, suppress=True)

size = 4
A = np.array(np.random.rand(size, size))

print("base matrix: ")
print(A)

L = np.zeros((size, size))
U = A

for i in range(size):
    L[i][i] = 1

for column in range(size - 1):
    for row in range(column + 1, size):

        factor = (-1) * U[row][column] / U[column][column]

        L[row][column] = (-1) * factor
        U[row] += factor * U[column]


def scale_matrix(matrix, size):

    scales = np.zeros((size, size))

    index = 0
    for row in matrix:

        max_index = 0

        for element in range(len(row)):
            if abs(row[element]) > abs(row[max_index]):
                max_index = element

        scales[index][index] = row[max_index]
        row /= row[max_index]

        index += 1

    return scales


print("\nL matrix:")
print(L)
print("\nU matrix:")
print(U)
print("\nL * U matrix:")
print(np.matmul(L, U))
Ls = scale_matrix(L, size)
print("\nL matrix scaled:")
print(L)
print("L scale matrix:")
print(Ls)
print("L scale matrix * L scaled:")
print(np.matmul(Ls, L))
Us = scale_matrix(U, size)
print("\nU matrix scaled:")
print(U)
print("U scale matrix * U scaled:")
print(Us)
print("U * U scale matrix:")
print(np.matmul(Us, U))
print("\nL scale matrix * L scaled *\n U scale matrix * U scaled:")
print(np.matmul(np.matmul(Ls, L), np.matmul(Us, U)))
