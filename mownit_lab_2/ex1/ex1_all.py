import numpy as np
import time

import ex1.ex1_diagonal_pivot
import ex1.ex1_partial_pivot
import ex1.ex1_full_pivot

m_size = 600
A = np.array(np.random.rand(m_size, m_size))
B = np.array(np.random.rand(m_size))

print("\nsize = " + str(m_size))

print("\nex 1: diagonal pivoting:\n")

A1 = A
B1 = B
exec_time = time.time()

ex1.ex1_diagonal_pivot.ex1_diagonal_pivot(m_size, A1, B1)

exec_time = time.time() - exec_time

print("execution time = " + str(exec_time))

print("\n\nex 1: partial pivoting:\n")

A2 = A
B2 = B
exec_time = time.time()

ex1.ex1_partial_pivot.ex1_partial_pivot(m_size, A2, B2)

exec_time = time.time() - exec_time

print("execution time = " + str(exec_time))

print("\n\nex 1: full pivoting:\n")

A3 = A
B3 = B
exec_time = time.time()

ex1.ex1_full_pivot.ex1_full_pivot(m_size, A3, B3)

exec_time = time.time() - exec_time

print("execution time = " + str(exec_time))

print("\n\nex 1: compare library function:\n")

A4 = A
B4 = B
exec_time = time.time()

np.linalg.solve(A4, B4)

exec_time = time.time() - exec_time

print("execution time = " + str(exec_time))