import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt
import time
import ex_1_pckg.cost_functions as cf


class Img:

    def __init__(self, size_y, size_x, density):

        self.size_y = size_y
        self.size_x = size_x
        self.energy = 0

        self.img = np.zeros((size_y, size_x))
        indexes = random.sample([(x, y) for x in range(size_y) for y in range(size_x)], int(size_y * size_x * density))

        for n, m in indexes:
            self.img[n][m] = 1

    def __copy__(self):
        result = Img(0, 0, 0.0)
        result.size_y = self.size_y
        result.size_x = self.size_x
        result.energy = self.energy
        result.img = self.img.copy()
        return result

    def calculate_energy(self):
        self.energy = 0

    def point_sum(self):
        return 0

    def save(self, name):
        Image.fromarray(np.uint8([[255.0 if e == 0 else 0.0 for e in a] for a in self.img])).save(name + '.png')

    def get_single_point_energy(self, y, x):
        return 0

    def get_point_swap_energy(self, y, x):
        return 0

    def calculate_swap_energy(self, swap):
        pass

    def execute_permutation(self, perm):
        for swap in perm:
            start_energy = self.calculate_swap_energy(swap)
            self.img[swap[0][0]][swap[0][1]], self.img[swap[1][0]][swap[1][1]] = \
                self.img[swap[1][0]][swap[1][1]], self.img[swap[0][0]][swap[0][1]]
            self.energy += self.calculate_swap_energy(swap) - start_energy


class Plus(Img):

    def get_single_point_energy(self, cy, cx):

        if cy >= self.size_y or cy < 0 or cx >= self.size_x or cx < 0:
            return 0

        sum = 0

        for y in range(-1, 2):
            for x in range(-1, 2):

                ny = cy + y
                if ny >= self.size_y or ny < 0:
                    break
                nx = cx + x
                if nx >= self.size_x or nx < 0:
                    break

                if y == 0 or x == 0:
                    if self.img[ny][nx] != self.img[cy][cx]:
                        sum += 1

                else:
                    if self.img[ny][nx] == self.img[cy][cx]:
                        sum += 1

        return sum

    def get_point_swap_energy(self, cy, cx):
        return 2 * self.get_single_point_energy(cy, cx)

    def calculate_swap_energy(self, swap):

        result = 0
        (y1, x1), (y2, x2) = swap
        result += self.get_point_swap_energy(y1, x1)
        result += self.get_point_swap_energy(y2, x2)
        # for y in range(-1, 2):
        #     for x in range(-1, 2):
        #         result += self.get_single_point_energy(y1 + y, x1 + x)
        #         result += self.get_single_point_energy(y2 + y, x2 + x)

        return result

    def calculate_energy(self):

        sum = 0

        for y in range(self.size_y):
            for x in range(self.size_x):
                sum += self.get_single_point_energy(y, x)

        self.energy = sum


class Inverted_plus(Img):

    def get_single_point_energy(self, cy, cx):

        if cy >= self.size_y or cy < 0 or cx >= self.size_x or cx < 0:
            return 0

        sum = 0

        for y in range(-1, 2):
            for x in range(-1, 2):

                ny = cy + y
                if ny >= self.size_y or ny < 0:
                    break
                nx = cx + x
                if nx >= self.size_x or nx < 0:
                    break

                if y == 0 or x == 0:
                    if self.img[ny][nx] == self.img[cy][cx]:
                        sum += 1

                else:
                    if self.img[ny][nx] != self.img[cy][cx]:
                        sum += 1

        return sum

    def get_point_swap_energy(self, cy, cx):
        return 2 * self.get_single_point_energy(cy, cx)

    def calculate_swap_energy(self, swap):

        result = 0
        (y1, x1), (y2, x2) = swap
        result += self.get_point_swap_energy(y1, x1)
        result += self.get_point_swap_energy(y2, x2)
        # for y in range(-1, 2):
        #     for x in range(-1, 2):
        #         result += self.get_single_point_energy(y1 + y, x1 + x)
        #         result += self.get_single_point_energy(y2 + y, x2 + x)

        return result

    def calculate_energy(self):

        sum = 0

        for y in range(self.size_y):
            for x in range(self.size_x):
                sum += self.get_single_point_energy(y, x)

        self.energy = sum


class Big_square(Img):

    def get_single_point_energy(self, cy, cx):

        if cy >= self.size_y or cy < 0 or cx >= self.size_x or cx < 0:
            return 0

        sum = 0

        for y in range(-2, 3, 1):
            for x in range(-2, 3, 1):

                ny = cy + y
                if ny >= self.size_y:
                    ny = ny % self.size_y
                nx = cx + x
                if nx >= self.size_x:
                    nx = nx % self.size_x

                if (y % 2 == 0 and y != 0) or (x % 2 == 0 and x != 0):
                    if self.img[ny][nx] != self.img[cy][cx]:
                        sum += 1

                else:
                    if self.img[ny][nx] == self.img[cy][cx]:
                        sum += 1

        return sum - 1

    def get_point_swap_energy(self, cy, cx):
        return 2 * self.get_single_point_energy(cy, cx)

    def calculate_swap_energy(self, swap):

        result = 0
        (y1, x1), (y2, x2) = swap
        result += self.get_point_swap_energy(y1, x1)
        result += self.get_point_swap_energy(y2, x2)

        return result

    def calculate_energy(self):

        sum = 0

        for y in range(self.size_y):
            for x in range(self.size_x):
                sum += self.get_single_point_energy(y, x)

        self.energy = sum


class Big_x(Img):

    def get_single_point_energy(self, cy, cx):

        if cy >= self.size_y or cy < 0 or cx >= self.size_x or cx < 0:
            return 0

        sum = 0

        for y in range(-2, 3, 1):
            for x in range(-2, 3, 1):

                ny = cy + y
                if ny >= self.size_y:
                    ny = ny % self.size_y
                nx = cx + x
                if nx >= self.size_x:
                    nx = nx % self.size_x

                if (abs(x) == abs(y)):
                    if self.img[ny][nx] != self.img[cy][cx]:
                        sum += 1

                else:
                    if self.img[ny][nx] == self.img[cy][cx]:
                        sum += 1

        return sum

    def get_point_swap_energy(self, cy, cx):
        return 2 * self.get_single_point_energy(cy, cx)

    def calculate_swap_energy(self, swap):

        result = 0
        (y1, x1), (y2, x2) = swap
        result += self.get_point_swap_energy(y1, x1)
        result += self.get_point_swap_energy(y2, x2)

        return result

    def calculate_energy(self):

        sum = 0

        for y in range(self.size_y):
            for x in range(self.size_x):
                sum += self.get_single_point_energy(y, x)

        self.energy = sum


class Corners(Img):

    def get_single_point_energy(self, cy, cx):

        if cy >= self.size_y or cy < 0 or cx >= self.size_x or cx < 0:
            return 0

        sum = 0

        for y in range(-2, 3, 1):
            for x in range(-2, 3, 1):

                ny = cy + y
                if ny >= self.size_y:
                    ny = ny % self.size_y
                nx = cx + x
                if nx >= self.size_x:
                    nx = nx % self.size_x

                if (abs(x) + abs(y) == 4):
                    if self.img[ny][nx] != self.img[cy][cx]:
                        sum += 1

                else:
                    if self.img[ny][nx] == self.img[cy][cx]:
                        sum += 1

        return sum - 1

    def get_point_swap_energy(self, cy, cx):
        return 2 * self.get_single_point_energy(cy, cx)

    def calculate_swap_energy(self, swap):

        result = 0
        (y1, x1), (y2, x2) = swap
        result += self.get_point_swap_energy(y1, x1)
        result += self.get_point_swap_energy(y2, x2)

        return result

    def calculate_energy(self):

        sum = 0

        for y in range(self.size_y):
            for x in range(self.size_x):
                sum += self.get_single_point_energy(y, x)

        self.energy = sum


class Boom(Img):

    # SHAPE:
    # x @ x @ x
    # @ x x x @
    # x x @ x x
    # @ x x x @
    # x @ x @ x

    def get_single_point_energy(self, cy, cx):

        if cy >= self.size_y or cy < 0 or cx >= self.size_x or cx < 0:
            return 0

        sum = 0

        for y in range(-2, 3, 1):
            for x in range(-2, 3, 1):

                ny = cy + y
                if ny >= self.size_y:
                    ny = ny % self.size_y
                nx = cx + x
                if nx >= self.size_x:
                    nx = nx % self.size_x

                if (abs(x) + abs(y) == 3):
                    if self.img[ny][nx] != self.img[cy][cx]:
                        sum += 1

                else:
                    if self.img[ny][nx] == self.img[cy][cx]:
                        sum += 1

        return sum - 1

    def get_point_swap_energy(self, cy, cx):
        return 2 * self.get_single_point_energy(cy, cx)

    def calculate_swap_energy(self, swap):

        result = 0
        (y1, x1), (y2, x2) = swap
        result += self.get_point_swap_energy(y1, x1)
        result += self.get_point_swap_energy(y2, x2)

        return result

    def calculate_energy(self):

        sum = 0

        for y in range(self.size_y):
            for x in range(self.size_x):
                sum += self.get_single_point_energy(y, x)

        self.energy = sum


class BoomCorners(Img):

    # SHAPE:
    # x @ x @ x
    # @ x x x @
    # x x @ x x
    # @ x x x @
    # x @ x @ x

    def get_single_point_energy(self, cy, cx):

        if cy >= self.size_y or cy < 0 or cx >= self.size_x or cx < 0:
            return 0

        sum = 0

        for y in range(-2, 3, 1):
            for x in range(-2, 3, 1):

                ny = cy + y
                if ny >= self.size_y:
                    ny = ny % self.size_y
                nx = cx + x
                if nx >= self.size_x:
                    nx = nx % self.size_x

                if abs(x) + abs(y) >= 3:
                    if self.img[ny][nx] != self.img[cy][cx]:
                        sum += 1

                else:
                    if self.img[ny][nx] == self.img[cy][cx]:
                        sum += 1

        return sum - 1

    def get_point_swap_energy(self, cy, cx):
        return 2 * self.get_single_point_energy(cy, cx)

    def calculate_swap_energy(self, swap):

        result = 0
        (y1, x1), (y2, x2) = swap
        result += self.get_point_swap_energy(y1, x1)
        result += self.get_point_swap_energy(y2, x2)

        return result

    def calculate_energy(self):

        sum = 0

        for y in range(self.size_y):
            for x in range(self.size_x):
                sum += self.get_single_point_energy(y, x)

        self.energy = sum


class BoldCorners(Img):

    # SHAPE:
    # @ @ x x x @ @
    # @ @ x x x @ @
    # x x x x x x x
    # x x x @ x x x
    # x x x x x x x
    # @ @ x x x @ @
    # @ @ x x x @ @

    def get_single_point_energy(self, cy, cx):

        if cy >= self.size_y or cy < 0 or cx >= self.size_x or cx < 0:
            return 0

        sum = 0

        for y in range(-3, 4, 1):
            for x in range(-3, 4, 1):

                ny = cy + y
                if ny >= self.size_y:
                    ny = ny % self.size_y
                nx = cx + x
                if nx >= self.size_x:
                    nx = nx % self.size_x

                if abs(x) + abs(y) >= 5 or (abs(x) == 2 and abs(y) == 2):
                    if self.img[ny][nx] != self.img[cy][cx]:
                        sum += 1

                else:
                    if self.img[ny][nx] == self.img[cy][cx]:
                        sum += 1

        return sum - 1

    def get_point_swap_energy(self, cy, cx):
        return 2 * self.get_single_point_energy(cy, cx)

    def calculate_swap_energy(self, swap):

        result = 0
        (y1, x1), (y2, x2) = swap
        result += self.get_point_swap_energy(y1, x1)
        result += self.get_point_swap_energy(y2, x2)

        return result

    def calculate_energy(self):

        sum = 0

        for y in range(self.size_y):
            for x in range(self.size_x):
                sum += self.get_single_point_energy(y, x)

        self.energy = sum


class InnerSquare(Img):

    # SHAPE:
    # @ @ @ x @ @ @
    # @ @ x x x @ @
    # @ x x x x x @
    # x x x @ x x x
    # @ x x x x x @
    # @ @ x x x @ @
    # @ @ @ x @ @ @

    def get_single_point_energy(self, cy, cx):

        if cy >= self.size_y or cy < 0 or cx >= self.size_x or cx < 0:
            return 0

        sum = 0

        for y in range(-3, 4, 1):
            for x in range(-3, 4, 1):

                ny = cy + y
                if ny >= self.size_y:
                    ny = ny % self.size_y
                nx = cx + x
                if nx >= self.size_x:
                    nx = nx % self.size_x

                if abs(x) + abs(y) >= 4:
                    if self.img[ny][nx] != self.img[cy][cx]:
                        sum += 1

                else:
                    if self.img[ny][nx] == self.img[cy][cx]:
                        sum += 1

        return sum - 1

    def get_point_swap_energy(self, cy, cx):
        return 2 * self.get_single_point_energy(cy, cx)

    def calculate_swap_energy(self, swap):

        result = 0
        (y1, x1), (y2, x2) = swap
        result += self.get_point_swap_energy(y1, x1)
        result += self.get_point_swap_energy(y2, x2)

        return result

    def calculate_energy(self):

        sum = 0

        for y in range(self.size_y):
            for x in range(self.size_x):
                sum += self.get_single_point_energy(y, x)

        self.energy = sum


class BoldShape(Img):

    # SHAPE:
    # @ @ @ @ @ @ @
    # @ @ @ @ @ @ @
    # @ @ x x x @ @
    # @ @ x @ x @ @
    # @ @ x x x @ @
    # @ @ @ @ @ @ @
    # @ @ @ @ @ @ @

    def get_single_point_energy(self, cy, cx):

        if cy >= self.size_y or cy < 0 or cx >= self.size_x or cx < 0:
            return 0

        sum = 0

        for y in range(-3, 4, 1):
            for x in range(-3, 4, 1):

                ny = cy + y
                if ny >= self.size_y:
                    ny = ny % self.size_y
                nx = cx + x
                if nx >= self.size_x:
                    nx = nx % self.size_x

                if abs(x) != 1 and abs(y) != 1:
                    if self.img[ny][nx] != self.img[cy][cx]:
                        sum += 1

                else:
                    if self.img[ny][nx] == self.img[cy][cx]:
                        sum += 1

        return sum

    def get_point_swap_energy(self, cy, cx):
        return 2 * self.get_single_point_energy(cy, cx)

    def calculate_swap_energy(self, swap):

        result = 0
        (y1, x1), (y2, x2) = swap
        result += self.get_point_swap_energy(y1, x1)
        result += self.get_point_swap_energy(y2, x2)

        return result

    def calculate_energy(self):

        sum = 0

        for y in range(self.size_y):
            for x in range(self.size_x):
                sum += self.get_single_point_energy(y, x)

        self.energy = sum


class ThreeSquares(Img):

    # SHAPE:
    # @ @ @ @ @ @ @
    # @ x x x x x @
    # @ x @ @ @ x @
    # @ x @ @ @ x @
    # @ x @ @ @ x @
    # @ x x x x x @
    # @ @ @ @ @ @ @

    def get_single_point_energy(self, cy, cx):

        if cy >= self.size_y or cy < 0 or cx >= self.size_x or cx < 0:
            return 0

        sum = 0

        for y in range(-3, 4, 1):
            for x in range(-3, 4, 1):

                ny = cy + y
                if ny >= self.size_y:
                    ny = ny % self.size_y
                nx = cx + x
                if nx >= self.size_x:
                    nx = nx % self.size_x

                if (abs(x) == 2 and abs(y) != 3) or (abs(y) == 2 and abs(x) != 3):
                    if self.img[ny][nx] == self.img[cy][cx]:
                        sum += 1

                else:
                    if self.img[ny][nx] != self.img[cy][cx]:
                        sum += 1

        return sum

    def get_point_swap_energy(self, cy, cx):
        return 2 * self.get_single_point_energy(cy, cx)

    def calculate_swap_energy(self, swap):

        result = 0
        (y1, x1), (y2, x2) = swap
        result += self.get_point_swap_energy(y1, x1)
        result += self.get_point_swap_energy(y2, x2)

        return result

    def calculate_energy(self):

        sum = 0

        for y in range(self.size_y):
            for x in range(self.size_x):
                sum += self.get_single_point_energy(y, x)

        self.energy = sum


class Symbol(Img):

    # SHAPE:
    # @ @ @ @ @ @ @
    # @ x x x x x @
    # @ x x @ x x @
    # @ x @ @ @ x @
    # @ x x @ x x @
    # @ x x x x x @
    # @ @ @ @ @ @ @

    def get_single_point_energy(self, cy, cx):

        if cy >= self.size_y or cy < 0 or cx >= self.size_x or cx < 0:
            return 0

        sum = 0

        for y in range(-3, 4, 1):
            for x in range(-3, 4, 1):

                ny = cy + y
                if ny >= self.size_y:
                    ny = ny % self.size_y
                nx = cx + x
                if nx >= self.size_x:
                    nx = nx % self.size_x

                if abs(x) == 3 or abs(y) == 3 or abs(x) + abs(y) == 1:
                    if self.img[ny][nx] != self.img[cy][cx]:
                        sum += 1

                else:
                    if self.img[ny][nx] == self.img[cy][cx]:
                        sum += 1

        return sum - 1

    def get_point_swap_energy(self, cy, cx):
        return 2 * self.get_single_point_energy(cy, cx)

    def calculate_swap_energy(self, swap):

        result = 0
        (y1, x1), (y2, x2) = swap
        result += self.get_point_swap_energy(y1, x1)
        result += self.get_point_swap_energy(y2, x2)

        return result

    def calculate_energy(self):

        sum = 0

        for y in range(self.size_y):
            for x in range(self.size_x):
                sum += self.get_single_point_energy(y, x)

        self.energy = sum


class SquareSplit(Img):

    # SHAPE:
    # @ @ @ @ @ @ @
    # @ x x @ x x @
    # @ x x @ x x @
    # @ @ @ @ @ @ @
    # @ x x @ x x @
    # @ x x @ x x @
    # @ @ @ @ @ @ @

    def get_single_point_energy(self, cy, cx):

        if cy >= self.size_y or cy < 0 or cx >= self.size_x or cx < 0:
            return 0

        sum = 0

        for y in range(-3, 4, 1):
            for x in range(-3, 4, 1):

                ny = cy + y
                if ny >= self.size_y:
                    ny = ny % self.size_y
                nx = cx + x
                if nx >= self.size_x:
                    nx = nx % self.size_x

                if abs(x) == 3 or abs(y) == 3 or abs(x) == 0 or abs(y) == 0:
                    if self.img[ny][nx] != self.img[cy][cx]:
                        sum += 1

                else:
                    if self.img[ny][nx] == self.img[cy][cx]:
                        sum += 1

        return sum

    def get_point_swap_energy(self, cy, cx):
        return 2 * self.get_single_point_energy(cy, cx)

    def calculate_swap_energy(self, swap):

        result = 0
        (y1, x1), (y2, x2) = swap
        result += self.get_point_swap_energy(y1, x1)
        result += self.get_point_swap_energy(y2, x2)

        return result

    def calculate_energy(self):

        sum = 0

        for y in range(self.size_y):
            for x in range(self.size_x):
                sum += self.get_single_point_energy(y, x)

        self.energy = sum


class Square45(Img):

    # SHAPE:
    # x x x @ x x x
    # x x @ x @ x x
    # x @ x x x @ x
    # @ x x @ x x @
    # x @ x x x @ x
    # x x @ x @ x x
    # x x x @ x x x

    def get_single_point_energy(self, cy, cx):

        if cy >= self.size_y or cy < 0 or cx >= self.size_x or cx < 0:
            return 0

        sum = 0

        for y in range(-3, 4, 1):
            for x in range(-3, 4, 1):

                ny = cy + y
                if ny >= self.size_y:
                    ny = ny % self.size_y
                nx = cx + x
                if nx >= self.size_x:
                    nx = nx % self.size_x

                if abs(x) + abs(y) == 3:
                    if self.img[ny][nx] != self.img[cy][cx]:
                        sum += 1

                else:
                    if self.img[ny][nx] == self.img[cy][cx]:
                        sum += 1

        return sum - 1

    def get_point_swap_energy(self, cy, cx):
        return 2 * self.get_single_point_energy(cy, cx)

    def calculate_swap_energy(self, swap):

        result = 0
        (y1, x1), (y2, x2) = swap
        result += self.get_point_swap_energy(y1, x1)
        result += self.get_point_swap_energy(y2, x2)

        return result

    def calculate_energy(self):

        sum = 0

        for y in range(self.size_y):
            for x in range(self.size_x):
                sum += self.get_single_point_energy(y, x)

        self.energy = sum


def generate_n_permutations(img, n, to_swap):

    permutations = []

    for x in range(n):

        black_indexes_raw = np.where(img.img == 1)
        black_indexes = list(zip(black_indexes_raw[0], black_indexes_raw[1]))

        white_indexes_raw = np.where(img.img == 0)
        white_indexes = list(zip(white_indexes_raw[0], white_indexes_raw[1]))

        changes = []
        for i in range(to_swap):
            changes.append((white_indexes.pop(random.randint(0, len(white_indexes) - 1)), black_indexes.pop(random.randint(0, len(black_indexes) - 1))))

        permutations.append(changes)

    return permutations


def get_possibility_to_change(curr_energy, new_energy, temperature, swaps):

    if new_energy < curr_energy:
        return 1.0

    if temperature == 0:
        return 0.0

    return np.exp((curr_energy - new_energy) / swaps / temperature)


def choose_neighbour(img, permutations, curr_energy, temperature, swaps):

    neighbours = []

    for perm in permutations:
        img.execute_permutation(perm)
        neighbours.append((perm, img.energy))
        img.execute_permutation(perm)

    neighbours = sorted(neighbours, key=lambda t: t[1])

    for perm, energy in neighbours:
        if get_possibility_to_change(curr_energy, energy, temperature, swaps) > random.random():
            return perm, energy

    return [((0, 0), (0, 0))], curr_energy


def loop(img, perm_nr, iterations, T_function, swap_number_function):

    best_img = img.__copy__()
    img.calculate_energy()
    best_energy = img.energy
    curr_energy = best_energy

    energies = [best_energy]
    T_list = []

    for i in range(iterations):

        print(i)

        T_list.append(T_function(i + 1))

        swaps = max(1, int(swap_number_function(i + 1)))
        permutations = generate_n_permutations(img, perm_nr, swaps)

        perm, curr_energy = choose_neighbour(img, permutations, curr_energy, T_list[i], swaps)
        img.execute_permutation(perm)

        if curr_energy < best_energy:
            best_energy = curr_energy
            best_img = img.__copy__()

        energies.append(curr_energy)

    print("---> ", best_energy)

    return best_img, energies, T_list


iterations = 15000
starting_T = 0.5
perm_nr = 1
size1 = 64
size2 = 64
swap_percent = 0.0005
fill_percent = 0.5

print(iterations, size1, size1, "new")

# img1 = Plus(size1, size2, fill_percent)
# img1 = Inverted_Plus(size1, size2, fill_percent)
# img1 = Big_square(size1, size2, fill_percent)
# img1 = Big_x(size1, size2, fill_percent)
# img1 = Corners(size1, size2, fill_percent)
# img1 = Boom(size1, size2, fill_percent)
# img1 = BoomCorners(size1, size2, fill_percent)
# img1 = BoldCorners(size1, size2, fill_percent)
# img1 = InnerSquare(size1, size2, fill_percent)
# img1 = BoldShape(size1, size2, fill_percent)
# img1 = ThreeSquares(size1, size2, fill_percent)
# img1 = Symbol(size1, size2, fill_percent)
# img1 = SquareSplit(size1, size2, fill_percent)
img1 = Square45(size1, size2, fill_percent)
img1.save('pre')

img1.calculate_energy()
print(img1.energy)
curr_time = time.time()
bit_arr, energies, T_list = loop(img1, perm_nr, iterations, cf.not_so_fast(iterations, starting_T),
    cf.quite_fast(iterations, max(1, int(size1 * size2 * swap_percent))))
print(" --- time --->", time.time() - curr_time)
img1.calculate_energy()
print(img1.energy)

x_axis = [i for i in range(len(energies))]
plt.plot(x_axis, energies)
plt.show()

T_x_axis = [i for i in range(len(T_list))]
plt.plot(T_x_axis, T_list)
plt.show()

img1.save('after')

