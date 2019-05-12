import numpy as np
import matplotlib.pyplot as mp
import time

print("1.1)")

N = int(1e7)

v = np.float32(0.53125)

time1 = time.time()

real_sum = N * v
sum1 = np.float32(0)
for i in range(N):
    sum1 += v

time1 = time.time() - time1

print("real sum: " + str(real_sum) + "\nsingle precision sum: " + str(sum1))

print("1.2)")

abs_err = abs(real_sum - sum1)
rel_err = abs_err / real_sum

print("absolute error: " + str(abs_err) + "\nrelative error: " + str(rel_err))
print("execution time: " + str(time1))

print("1.3)")


def ex1pt3(N, value):
    step = int(25000)

    x_axis = np.arange(0.0, N / step, 1.0)
    y_axis = np.zeros(x_axis.size)
    sum_step = np.float32(0)
    for i in range(y_axis.size):
        for j in range(step):
            sum_step += value
        real_sum_step = i * step * value
        if i != 0:
            y_axis[i] = abs(real_sum_step - sum_step) / real_sum_step
    y_axis[0] = y_axis[1]

    mp.plot(x_axis, y_axis)
    mp.show()


ex1pt3(N, v)

print("1.4)")


def recur(values):
    if values.size == 1:
        return values
    if values.size % 2 == 1:
        result = np.zeros(int(values.size / 2 + 1))
        for x in range(0, values.size - 1, 2):
            result[int(x / 2)] = values[x] + values[x + 1]
        result[-1] = values[-1]
    else:
        result = np.zeros(int(values.size / 2))
        for x in range(0, values.size, 2):
            result[int(x / 2)] = np.float32(values[x] + values[x + 1])
    return recur(result)


def ex1pt4(N, value):
    start_arr = np.full((N, 1), value)
    return recur(start_arr)


time2 = time.time()

sum_recur = ex1pt4(N, v)

time2 = time.time() - time2

abs_err_recur = abs(real_sum - sum_recur)
rel_err_recur = abs_err_recur / real_sum

print("1.5)")

print("absolute error: " + str(abs_err_recur) + "\nrelative error: " + str(rel_err_recur))
print("execution time: " + str(time2))

print("1.6)")
print("time_recur / time_simple = " + str(time2 / time1))

print("1.7)")


def ex1pt7(N, values):
    start_arr = np.zeros(N)
    for j in range(N):
        start_arr[j] = values[j % len(values)]
    return recur(start_arr)


fill_values = [np.float32(0.10010), np.float32(0.899899)]

time2_2 = time.time()

sum_recur_2 = ex1pt7(N, fill_values)

time2_2 = time.time() - time2_2

abs_err_recur_2 = abs(real_sum - sum_recur_2)
rel_err_recur_2 = abs_err_recur_2 / real_sum

print("absolute error: " + str(abs_err_recur_2) + "\nrelative error: " + str(rel_err_recur_2))
print("execution time: " + str(time2_2))

print("2.1)")


def kahan(values):
    kahan_sum = np.float32(0)
    kahan_err = np.float32(0)
    for i in range(values.size):
        y = values[i] - kahan_err
        tmp = kahan_sum + y
        kahan_err = tmp - kahan_sum - y
        kahan_sum = tmp
    return kahan_sum


def ex2(N, value):
    start_arr = np.full((N, 1), value)
    return kahan(start_arr)


time3 = time.time()

sum_kahan = ex2(N, v)

time3 = time.time() - time3

abs_err_kahan = abs(real_sum - sum_kahan)
rel_err_kahan = abs_err_kahan / real_sum

print("absolute error: " + str(abs_err_kahan) + "\nrelative error: " + str(rel_err_kahan))
print("execution time: " + str(time3))

print("2.2)")

print("2.3)")
print("time_kahan / time_recur = " + str(time3 / time2))


