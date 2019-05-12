import numpy as np


def fast(all_iter, starting_T):
    return lambda x: 1 / (x ** (1 / (((-4 / all_iter) * x) + 5))) * starting_T


def slow_linear(all_iter, starting_T):
    return lambda x: (-starting_T) / all_iter * x + starting_T


def quite_fast_leaves_chance(all_iter, starting_T):
    return lambda x: np.exp(-x / (all_iter / 3)) * starting_T


def quite_fast(all_iter, starting_T):
    return lambda x: np.sin(-np.sin(x * np.pi / 2 / all_iter) * np.pi / 2) * starting_T + starting_T


def not_so_fast(all_iter, starting_T):
    return lambda x: -np.sin(x * np.pi / 2 / all_iter) * starting_T + starting_T