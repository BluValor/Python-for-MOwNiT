import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt
import ex_2_pckg.old.neighbour_functions as nf
import ex_1_pckg.cost_functions as cf
import time


def generate_random_bit_array(size_y, size_x, density):

    result = np.zeros((size_y, size_x))
    indexes = random.sample([(x, y) for x in range(size_y) for y in range(size_x)], int(size_y * size_x * density))

    for n, m in indexes:
        result[n][m] = 1

    return result


def prepare_image_to_show(bit_arr):
    return [[255.0 if e == 0 else 0.0 for e in a] for a in bit_arr]


# def count_energy_no_edges(image, coords, fun):
#
#     mid_y, mid_x = np.shape(fun())[0] // 2, np.shape(fun())[1] // 2
#     height, lenght = np.shape(image)
#     sum = 0
#
#     for y, row in enumerate(fun()):
#         for x, type in enumerate(row):
#
#             neighbour_y, neighbour_x = coords[0] + y - mid_y, coords[1] + x - mid_x
#
#             if neighbour_y < 0: neighbour_y = height + neighbour_y
#             if neighbour_y >= height: neighbour_y = neighbour_y - height
#             if neighbour_x < 0: neighbour_x = lenght + neighbour_x
#             if neighbour_x >= lenght: neighbour_x = neighbour_x - lenght
#
#             neighbour_type = nf.Color(image[neighbour_y][neighbour_x])
#
#             if type is neighbour_type or type is nf.Color.NONE or type is nf.Color.SELF:
#                 pass
#             else:
#                 sum += 1
#
#     print("-> ", sum)
#     return sum


# def count_energy_w_edges(image, coords, fun):
#
#     mid_y, mid_x = np.shape(fun())[0] // 2, np.shape(fun())[1] // 2
#     height, lenght = np.shape(image)
#     sum = 0
#
#     for y, row in enumerate(fun()):
#         for x, type in enumerate(row):
#
#             neighbour_y, neighbour_x = coords[0] + y - mid_y, coords[1] + x - mid_x
#
#             if not (neighbour_y < 0 or neighbour_y >= height or neighbour_x < 0 or neighbour_x >= lenght):
#
#                 neighbour_type = nf.Color(image[neighbour_y][neighbour_x])
#
#                 if type is neighbour_type or type is nf.Color.NONE or type is nf.Color.SELF:
#                     pass
#                 else:
#                     sum += 1
#
#     return sum


def calculate_image_energy(image, fun, y_shape, x_shape):

    sum = 0

    y = 0
    for row in image:
        for x in range(len(row)):
            sum += fun(image, (y, x), y_shape, x_shape)
        y += 1

    return sum


def generate_n_permutations(image, n, to_swap):

    permutations = []

    for x in range(n):

        black_indexes_raw = np.where(image == 1)
        black_indexes = list(zip(black_indexes_raw[0], black_indexes_raw[1]))

        white_indexes_raw = np.where(image == 0)
        white_indexes = list(zip(white_indexes_raw[0], white_indexes_raw[1]))

        changes = []
        for i in range(to_swap):
            changes.append((white_indexes.pop(random.randint(0, len(white_indexes) - 1)), black_indexes.pop(random.randint(0, len(black_indexes) - 1))))

        permutations.append(changes)

    return permutations


def execute_permutation(image, perm):
    for swap in perm:
        image[swap[0][0]][swap[0][1]], image[swap[1][0]][swap[1][1]] = image[swap[1][0]][swap[1][1]], image[swap[0][0]][swap[0][1]]


def get_possibility_to_change(curr_energy, new_energy, temperature, swaps):

    if new_energy < curr_energy:
        return 1.0

    if temperature == 0:
        return 0.0

    return np.exp((curr_energy - new_energy) / swaps / temperature)


def choose_neighbour(image, permutations, curr_energy, temperature, fun, y_shape, x_shape, swaps):

    neighbours = []

    for perm in permutations:
        tmp = image.copy()
        execute_permutation(tmp, perm)
        neighbours.append((perm, calculate_image_energy(tmp, fun, y_shape, x_shape)))

    neighbours = sorted(neighbours, key=lambda t: t[1])

    for perm, energy in neighbours:
        if get_possibility_to_change(curr_energy, energy, temperature, swaps) > random.random():
            return perm, energy

    return [((0, 0), (0, 0))], curr_energy


def loop(image, perm_nr, iterations, T_function, neighbour_fun, swap_number_function):

    y_shape, x_shape = np.shape(image)

    best_image = image.copy()
    best_energy = calculate_image_energy(image, neighbour_fun, y_shape, x_shape)
    curr_energy = best_energy

    energies = [best_energy]
    T_list = []

    for i in range(iterations):

        T_list.append(T_function(i + 1))

        swaps = max(1, int(swap_number_function(i + 1)))
        permutations = generate_n_permutations(image, perm_nr, swaps)

        perm, curr_energy = choose_neighbour(image, permutations, curr_energy, T_list[i], neighbour_fun, y_shape, x_shape, swaps)
        execute_permutation(image, perm)

        if curr_energy < best_energy:
            best_energy = curr_energy
            best_image = image.copy()

        energies.append(curr_energy)

    return best_image, energies, T_list


iterations = 500
starting_T = 0.5
perm_nr = 1
size1 = 100
size2 = 100
swap_percent = 0.005
fill_percent = 0.5
prefix = ''

bit_arr = generate_random_bit_array(size1, size2, fill_percent)

show_arr = prepare_image_to_show(bit_arr)
im = Image.fromarray(np.uint8(show_arr))
# Image._show(im)
im.save(prefix + '_pre.png')

print(iterations, size1, size2)

curr_time = time.time()
bit_arr, energies, T_list = loop(bit_arr, perm_nr, iterations, cf.not_so_fast(iterations, starting_T),
    nf.no_edges_same_colour_plus, cf.quite_fast(iterations, max(1, int(size1 * size2 * swap_percent))))
print(" --- time --->", time.time() - curr_time)

x_axis = [i for i in range(len(energies))]
plt.plot(x_axis, energies)
plt.show()

T_x_axis = [i for i in range(len(T_list))]
plt.plot(T_x_axis, T_list)
plt.show()

show_arr = prepare_image_to_show(bit_arr)
im = Image.fromarray(np.uint8(show_arr))
# Image._show(im)
im.save(prefix + '_after.png')

