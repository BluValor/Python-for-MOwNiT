import numpy as np
import matplotlib.pyplot as mp

print("4)")

lower_x = 0
upper_x = 1

start_r = 3.6543
start_x = 0.6

start_N = 2000


def recursion(max_n, r, x):
    result = x
    for i in range(max_n):
        result = r * result * (1 - result)
    return result


def ex4(max_n, r_arr, x0):
    result = np.zeros(len(r_arr))
    for i in range(len(r_arr)):
        result[i] = recursion(max_n, r_arr[i], x0)
    return result


def ex4_a():
    x_array = np.arange(0.05, 0.95, 0.05)
    r_low = 1.0
    r_up = 4.0
    r_n = 5000
    r_iterations = 1000
    final_r_array = []
    final_r_n_array = []
    for i in range(len(x_array)):
        r_array = np.arange(r_low, r_up, (r_up - r_low) / r_n)
        r_n_array = ex4(r_iterations, r_array, x_array[i])
        final_r_array.extend(r_array)
        final_r_n_array.extend(r_n_array)
    mp.xlabel("r")
    mp.ylabel("x_n")
    mp.scatter(final_r_array, final_r_n_array, 0.1)
    mp.show()


ex4_a()


def recursion_32(max_n, r, x):
    result = np.float32(x)
    for i in range(max_n):
        result = np.float32(r) * result * (np.float32(1) - result)
    return result


def ex4_32(max_n, r_arr, x0):
    result = np.zeros(len(r_arr))
    for i in range(len(r_arr)):
        result[i] = np.float32(recursion(max_n, r_arr[i], x0))
    return result


def ex4_b_32():
    x_array = np.arange(0.05, 0.95, 0.05)
    r_low = 3.75
    r_up = 3.8
    r_n = 5000
    r_iterations = 1000
    final_r_array = []
    final_r_n_array = []
    for i in range(len(x_array)):
        r_array = np.arange(r_low, r_up, (r_up - r_low) / r_n)
        r_n_array = ex4(r_iterations, r_array, x_array[i])
        final_r_array.extend(r_array)
        final_r_n_array.extend(r_n_array)
    mp.xlabel("r")
    mp.ylabel("x_n")
    mp.scatter(final_r_array, final_r_n_array, 0.1)
    mp.show()


def recursion_64(max_n, r, x):
    result = np.float64(x)
    for i in range(max_n):
        result = np.float64(r) * result * (np.float64(1) - result)
    return result


def ex4_64(max_n, r_arr, x0):
    result = np.zeros(len(r_arr))
    for i in range(len(r_arr)):
        result[i] = np.float64(recursion(max_n, r_arr[i], x0))
    return result


def ex4_b_64():
    x_array = np.arange(0.05, 0.95, 0.05)
    r_low = 3.75
    r_up = 3.8
    r_n = 5000
    r_iterations = 1000
    final_r_array = []
    final_r_n_array = []
    for i in range(len(x_array)):
        r_array = np.arange(r_low, r_up, (r_up - r_low) / r_n)
        r_n_array = ex4(r_iterations, r_array, x_array[i])
        final_r_array.extend(r_array)
        final_r_n_array.extend(r_n_array)
    mp.xlabel("r")
    mp.ylabel("x_n")
    mp.scatter(final_r_array, final_r_n_array, 0.1)
    mp.show()


ex4_b_32()

ex4_b_64()


def ex4_c_r_count(r, x, N):
    curr_x = np.float32(x)
    results = [np.zeros(N), np.zeros(N)]
    results[1][0] = x
    for i in range(N):
        curr_x =np.float32(np.float32(r) * curr_x * (np.float32(1) - curr_x))
        results[0][i] = i
        results[1][i] = curr_x
    return results


def ex4_c_show():
    N = 100000
    x_array = [0.235, 0.2134, 0.6543, 0.346, 0.86545, 0.25545, 0.5432, 0.856, 0.453, 0.5863, 0.231, 0.1123]
    r = 4.0
    n_final = []
    x_final = []
    for xi in x_array:
        part = ex4_c_r_count(r, xi, N)
        n_final.extend(part[0])
        x_final.extend(part[1])
        zer0_found = False
        zero_index = 0
        while not zer0_found:
            if part[1][zero_index] - 1e-10 < 0:
                zer0_found = True
                print(str(xi) + " - " + str(zero_index))
            zero_index += 1
            if (zero_index >= N):
                zer0_found = True
    mp.xlabel("n")
    mp.ylabel("x_n")
    mp.scatter(n_final, x_final, 0.1)
    mp.show()


ex4_c_show()
