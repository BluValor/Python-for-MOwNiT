import numpy as np
import scipy
from scipy.linalg import decomp_lu as slu


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def make_unitary_vector(l):
    v = np.random.rand(l, 1)
    return v / np.linalg.norm(v)


def power_iterations(A, v, epsilon=1e-16, max_iter=1000):

    prev = v
    w = A @ v
    v = w / np.linalg.norm(w)

    i = 0
    while np.linalg.norm(prev - v) > epsilon and i < max_iter:

        prev = v
        w = A @ v
        v = w / np.linalg.norm(w)
        i += 1

    return np.transpose(v) @ A @ v


def increase_power_iterations(A, v, epsilon=1e-16, max_iter=10000, start_u=1000, mult=0.8, k=5):

    result = []
    u = start_u

    for _ in range(k):

        LU = slu.lu_factor(np.linalg.inv(A - np.diag([u for _ in range(len(A))])))
        prev = v
        w = slu.lu_solve(LU, v)
        v = w / np.linalg.norm(w)

        i = 0
        while np.linalg.norm(prev - v) > epsilon and i < max_iter:

            prev = v
            w = slu.lu_solve(LU, v)
            v = w / np.linalg.norm(w)
            i += 1

        result.append((np.transpose(v) @ A @ v)[0][0])
        u *= mult

    return result


l = 5


A_start = np.random.rand(l, l)
A = A_start @ np.transpose(A_start)

v = make_unitary_vector(l)

# power iterations

# expected = np.linalg.eigvals(A)[0]
# for i in [1, 2, 5, 10, 100, 200, 500, 1000, 2000, 10000]:
#     v = make_unitary_vector(l)
#     obtained = power_iterations(A, v, max_iter=i)[0][0]
#     print('\niterations:', i, '\nobtained:', obtained, ', expected:', expected,
#           ', difference:', np.abs(obtained - expected))

# increase power iterations

print(increase_power_iterations(A, v))
print(np.linalg.eigvals(A))


