import numpy as np
import random
import time
import matplotlib.pyplot as plt
import ex_1_pckg.cost_functions as cf


class Sudoku:

    def __init__(self, puzzle):

        self.puzzle = puzzle

        self.empty = []
        for y in range(9):
            for x in range(9):
                if self.puzzle[y][x] == 0:
                    self.empty.append((y, x))

    def __str__(self):

        result = "/-----------------------\\\n"

        for y in range(9):

            if y % 3 == 0 and y != 0:
                result += '|-------+-------+-------|\n'

            result += '| '

            for x in range(9):

                if x % 3 == 0 and x != 0:
                    result += '| '

                result += str(self.puzzle[y][x]) + ' '

            result += '|\n'

        return result + '\\-----------------------/'

    def __copy__(self):
        result = Sudoku(self.puzzle.copy())
        result.empty = self.empty.copy()
        return result

    def calculate_energy(self):

        sum = 0

        for y in range(3):
            for x in range(3):

                numbers = []

                for yoff in range(3):
                    for xoff in range(3):

                        if self.puzzle[y * 3 + yoff][x * 3 + xoff] in numbers:
                            sum += 1
                        else:
                            numbers.append(self.puzzle[y * 3 + yoff][x * 3 + xoff])

        # In order to simplify and speed up algorithm, when filling empty spaces in matrix I already take care of one
        # of the conditions - elements can not repeat in rows. Due to this I can neglect checking this condition
        # when counting the energy of the layout.

        # for y in range(9):
        #
        #     numbers = []
        #
        #     for x in range(9):
        #
        #         if self.puzzle[y][x] in numbers:
        #             sum += 1
        #         else:
        #             numbers.append(self.puzzle[y][x])

        for x in range(9):

            numbers = []

            for y in range(9):

                if self.puzzle[y][x] in numbers:
                    sum += 1
                else:
                    numbers.append(self.puzzle[y][x])

        return sum

    def is_in_row(self, number, row):

        for i in range(9):
            if self.puzzle[row][i] == number:
                return True
        return False

    def re_empty(self):
        for coords in self.empty:
            self.puzzle[coords] = 0

    def random_fill_empty(self):

        for tup in self.empty:

            number = random.randint(1, 9)
            while self.is_in_row(number, tup[0]):
                number = random.randint(1, 9)

            self.puzzle[tup] = number

    def swap(self, swap_tuple):
        row, column1, column2 = swap_tuple
        self.puzzle[row, column1], self.puzzle[row, column2] = self.puzzle[row, column2], self.puzzle[row, column1]


def parse_sudoku(list_input):

    quiz = np.zeros(81, np.int32)

    for y, sub_list in enumerate(list_input):
        for x, elem in enumerate(sub_list):
            quiz[9 * y + x] = elem

    quiz = quiz.reshape((9, 9))
    return Sudoku(quiz)


def get_rand_sudoku_from_file(amount):

    index = random.randint(0, amount - 1)

    quizzes = np.zeros((amount, 81), np.int32)
    solutions = np.zeros((amount, 81), np.int32)

    for i, line in enumerate(open('sudoku.csv', 'r').read().splitlines()[1:amount + 1]):
        quiz, solution = line.split(",")
        for j, q_s in enumerate(zip(quiz, solution)):
            q, s = q_s
            quizzes[i, j] = q
            solutions[i, j] = s

    quizzes = quizzes.reshape((-1, 9, 9))
    # solutions = solutions.reshape((-1, 9, 9))
    return Sudoku(quizzes[index])


def generate_0_n_square_permutations_more_random(sudoku, n):

    repeating = [[], [], [], [], [], [], [], [], []]

    for i in range(9):

        numbers = {}

        y_square = i // 3
        x_square = i % 3

        for y in range(3):
            for x in range(3):
                number = sudoku.puzzle[y_square * 3 + y][x_square * 3 + x]

                if number not in numbers:
                    numbers[number] = (y_square * 3 + y, x_square * 3 + x)
                else:
                    repeating[y_square * 3 + y].append(x_square * 3 + x)

    permutations = []
    for nr, lst in enumerate(repeating):
        while lst:
            elem = lst.pop()
            sec_elem = random.randint(0, 8)
            while sec_elem == elem:
                sec_elem = random.randint(0, 8)
            permutations.append((nr, elem, sec_elem))

    random.shuffle(permutations)
    return permutations[0:n]


def generate_0_n_square_permutations(sudoku, n):

    repeating = [[], [], [], [], [], [], [], [], []]

    for i in range(9):

        numbers = {}

        y_square = i // 3
        x_square = i % 3

        for y in range(3):
            for x in range(3):
                number = sudoku.puzzle[y_square * 3 + y][x_square * 3 + x]

                if number not in numbers:
                    numbers[number] = (y_square * 3 + y, x_square * 3 + x)
                else:
                    if number not in repeating[numbers[number][0]]:
                        repeating[numbers[number][0]].append(numbers[number][1])
                    repeating[y_square * 3 + y].append(x_square * 3 + x)

    permutations = []
    for nr, lst in enumerate(repeating):
        random.shuffle(lst)
        while lst:
            elem = lst.pop()
            if lst:
                permutations.append((nr, elem, lst.pop()))

    random.shuffle(permutations)
    return permutations[0:n]


def generate_0_n_vertical_permutations_more_random(sudoku, n):

    repeating = [[], [], [], [], [], [], [], [], []]

    for x in range(9):

        numbers = {}

        for y in range(9):

            number = sudoku.puzzle[y][x]

            if number not in numbers:
                numbers[number] = y
            else:
                repeating[y].append(x)

    permutations = []
    for nr, lst in enumerate(repeating):
        while lst:
            elem = lst.pop()
            sec_elem = random.randint(0, 8)
            while sec_elem == elem:
                sec_elem = random.randint(0, 8)
            permutations.append((nr, elem, sec_elem))

    random.shuffle(permutations)
    return permutations[0:n]


def generate_0_n_vertical_permutations(sudoku, n):

    repeating = [[], [], [], [], [], [], [], [], []]

    for x in range(9):

        numbers = {}

        for y in range(9):

            number = sudoku.puzzle[y][x]

            if number not in numbers:
                numbers[number] = y
            else:
                if number not in repeating[y]:
                    repeating[numbers[number]].append(x)
                repeating[y].append(x)

    permutations = []
    for nr, lst in enumerate(repeating):
        random.shuffle(lst)
        while lst:
            elem = lst.pop()
            if lst:
                permutations.append((nr, elem, lst.pop()))

    random.shuffle(permutations)
    return permutations[0:n]


def generate_n_permutations(sudoku, n):

    if random.random() > 0.3333:
        permutations = generate_0_n_square_permutations_more_random(sudoku, n)
    elif random.random() > 0.5:
        permutations = generate_0_n_vertical_permutations_more_random(sudoku, n)
    else:
        permutations = []

    while len(permutations) != n:

        row = random.randint(1, 9)
        indexes = [tup[1] for tup in sudoku.empty if tup[0] == row]
        if len(indexes) != 0:
            permutations.append(tuple([row] + random.sample(indexes, 2)))

    return permutations


def get_possibility_to_change(curr_energy, new_energy, temperature):

    if new_energy < curr_energy:
        return 1.0

    if temperature == 0:
        return 0.0

    return np.sin(np.exp((curr_energy - new_energy) / temperature) * np.pi/2)


def choose_neighbour(sudoku, permutations, curr_energy, temperature):

    neighbours = []

    for perm in permutations:
        sudoku.swap(perm)
        neighbours.append((perm, sudoku.calculate_energy()))
        sudoku.swap(perm)

    neighbours = sorted(neighbours, key=lambda t: t[1])

    for perm, energy in neighbours:
        if get_possibility_to_change(curr_energy, energy, temperature) > random.random():
            return perm, energy

    return (0, 0, 0), curr_energy


def loop(sudoku, perm_nr, iterations, starting_T, T_function_template):

    T_function = T_function_template(iterations, starting_T)

    best_attempt = sudoku.__copy__()
    best_energy = sudoku.calculate_energy()
    curr_energy = best_energy

    energies = [best_energy]
    T_list = []

    for i in range(iterations):

        T_list.append(T_function(i + 1))
        permutations = generate_n_permutations(sudoku, perm_nr)

        perm, curr_energy = choose_neighbour(sudoku, permutations, curr_energy, T_list[i])
        sudoku.swap(perm)

        if curr_energy < best_energy:
            best_energy = curr_energy
            best_attempt = sudoku.__copy__()

        energies.append(curr_energy)

    return best_attempt, energies, T_list


def consecutive_loop(sudoku, perm_nr, iterations, starting_T, T_function_template, max_attempts):

    best_attempt = sudoku.__copy__()
    best_energy = sudoku.calculate_energy()

    energies = [best_energy]
    T_list = []

    for i in range(max_attempts):

        sudoku_alt, energies_plus, T_list_plus = loop(sudoku, perm_nr, iterations, min(1.0, starting_T + 0.03),
                                                      T_function_template)
        energies += energies_plus
        T_list += T_list_plus
        alt_energy = sudoku_alt.calculate_energy()

        if alt_energy < best_energy:
            best_attempt = sudoku_alt
            best_energy = alt_energy

        if best_energy == 0:
            return best_attempt, energies, T_list

        sudoku.re_empty()
        sudoku.random_fill_empty()

    return best_attempt, energies, T_list


iterations = 5000
starting_T = 0.75
perm_nr = 3

max_attempts = 1

sudoku = get_rand_sudoku_from_file(1000)

# sudoku = [[9, 0, 0, 0, 0, 0, 0, 0, 0],
#           [0, 0, 3, 6, 0, 0, 0, 0, 0],
#           [0, 7, 0, 0, 9, 0, 2, 0, 0],
#           [0, 5, 0, 0, 0, 7, 0, 0, 0],
#           [0, 0, 0, 0, 4, 5, 7, 0, 0],
#           [0, 0, 0, 1, 0, 0, 0, 3, 0],
#           [0, 0, 1, 0, 0, 0, 0, 6, 8],
#           [0, 0, 8, 5, 0, 0, 0, 1, 0],
#           [0, 9, 0, 0, 0, 0, 4, 0, 0]]

# sudoku = [[0, 0, 5, 3, 0, 0, 0, 0, 0],
#           [8, 0, 0, 0, 0, 0, 0, 2, 0],
#           [0, 7, 0, 0, 1, 0, 5, 0, 0],
#           [4, 0, 0, 0, 0, 5, 3, 0, 0],
#           [0, 1, 0, 0, 7, 0, 0, 0, 6],
#           [0, 0, 3, 2, 0, 0, 0, 8, 0],
#           [0, 6, 0, 5, 0, 0, 0, 0, 9],
#           [0, 0, 4, 0, 0, 0, 0, 3, 0],
#           [0, 0, 0, 0, 0, 9, 7, 0, 0]]

sudoku = [[8, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 3, 6, 0, 0, 0, 0, 0],
          [0, 7, 0, 0, 9, 0, 2, 0, 0],
          [0, 5, 0, 0, 0, 7, 0, 0, 0],
          [0, 0, 0, 0, 4, 5, 7, 0, 0],
          [0, 0, 0, 1, 0, 0, 0, 3, 0],
          [0, 0, 1, 0, 0, 0, 0, 6, 8],
          [0, 0, 8, 5, 0, 0, 0, 1, 0],
          [0, 9, 0, 0, 0, 0, 4, 0, 0]]

sudoku = parse_sudoku(sudoku)

print("before:")
print(str(sudoku).replace('0', 'x'))
sudoku.random_fill_empty()

curr_time = time.time()
sudoku, energies, T_list = loop(sudoku, perm_nr, iterations, starting_T, cf.slow_linear)
# sudoku, energies, T_list = consecutive_loop(sudoku, perm_nr, iterations, starting_T, cf.slow_linear, max_attempts)
print("\n --- time --->", time.time() - curr_time, "\n")

x_axis = [i for i in range(len(energies))]
plt.plot(x_axis, energies)
plt.show()

T_x_axis = [i for i in range(len(T_list))]
plt.plot(T_x_axis, T_list)
plt.show()

print("after:")
print(sudoku)
print("layout energy: ", sudoku.calculate_energy())
