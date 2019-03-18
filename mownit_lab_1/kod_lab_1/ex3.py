import numpy as np

print("3)")

s = [2, 3.6667, 5, 7.2, 10]
n = [50, 100, 200, 500, 1000]


def dzeta_f_single(s, n):
    total_sum = np.float32(0)
    for k in range(1, n + 1):
        total_sum += np.float32(1) / np.float32(k) ** np.float32(s)
    return total_sum


def dzeta_b_single(s, n):
    total_sum = np.float32(0)
    for k in range(n, 0, -1):
        total_sum += np.float32(1) / np.float32(k) ** np.float32(s)
    return total_sum


def dzeta_f_double(s, n):
    total_sum = np.float64(0)
    for k in range(1, n + 1):
        total_sum += np.float64(1) / np.float64(k) ** np.float64(s)
    return total_sum


def dzeta_b_double(s, n):
    total_sum = np.float64(0)
    for k in range(n, 0, -1):
        total_sum += np.float64(1) / np.float64(k) ** np.float64(s)
    return total_sum


print("\n\tDZETA:\n")

for sx in s[::]:
    for nx in n[::]:
        print("s = " + str(sx) + ", n = " + str(nx) + ":"
              + "\ndzeta single precision forward:  " + str(dzeta_f_single(sx, nx))
              + "\ndzeta single precision backward: " + str(dzeta_b_single(sx, nx))
              + "\ndzeta double precision forward:  " + str(dzeta_f_double(sx, nx))
              + "\ndzeta double precision backward: " + str(dzeta_b_double(sx, nx))
              + "\n")


def eta_f_single(s, n):
    total_sum = np.float32(0)
    for k in range(1, n + 1):
        total_sum += (np.float32(-1) ** (np.float32(k) - np.float32(1))) * np.float32(1) / (np.float32(k) ** np.float32(s))
    return total_sum


def eta_b_single(s, n):
    total_sum = np.float32(0)
    for k in range(n, 0, -1):
        total_sum += (np.float32(-1) ** (np.float32(k) - np.float32(1))) * np.float32(1) / (np.float32(k) ** np.float32(s))
    return total_sum


def eta_f_double(s, n):
    total_sum = np.float64(0)
    for k in range(1, n + 1):
        total_sum += (np.float64(-1) ** (np.float64(k) - np.float64(1))) * np.float64(1) / (np.float64(k) ** np.float64(s))
    return total_sum


def eta_b_double(s, n):
    total_sum = np.float64(0)
    for k in range(n, 0, -1):
        total_sum += (np.float64(-1) ** (np.float64(k) - np.float64(1))) * np.float64(1) / (np.float64(k) ** np.float64(s))
    return total_sum


print("\n\tETA:\n")

for sx in s[::]:
    for nx in n[::]:
        print("s = " + str(sx) + ", n = " + str(nx) + ":"
              + "\neta single precision forward:  " + str(eta_f_single(sx, nx))
              + "\neta single precision backward: " + str(eta_b_single(sx, nx))
              + "\neta double precision forward:  " + str(eta_f_double(sx, nx))
              + "\neta double precision backward: " + str(eta_b_double(sx, nx))
              + "\n")
