import numpy as np
import matplotlib.pyplot as mp

epsilon = 1e-10

function_f1 = lambda x: np.cos(x) * np.cosh(x) - 1
borders_f1 = [1.5 * np.pi, 2 * np.pi]


def derivative(func, x):
    h = 1e-5
    return (func(x + h) - func(x)) / h


def bisection(func, borders):
    found = False
    result = 0.0
    iteration = 0
    while not found:
        iteration += 1
        middle = ((borders[0] + borders[1]) / 2)
        if abs(func(middle)) < epsilon:
            found = True
            result = middle
        if func(middle) * func(borders[0]) < 0:
            borders[1] = middle
        else:
            borders[0] = middle
    return [result, iteration]


# print(bisection(function_f1, borders_f1))

# x = np.linspace(1.5 * np.pi, 2 * np.pi, 10000)
# mp.plot(x, function_f1(x))
# mp.show()


def newton(func, borders):
    found = False
    result = 0.0
    iteration = 0
    middle = ((borders[0] + borders[1]) / 2)
    while not found:
        iteration += 1
        middle = middle - (function_f1(middle) / derivative(function_f1, middle))
        if abs(func(middle)) < epsilon:
            found = True
            result = middle
        if func(middle) * func(borders[0]) < 0:
            borders[1] = middle
        else:
            borders[0] = middle
    return [result, iteration]

# print(newton(function_f1, borders_f1))