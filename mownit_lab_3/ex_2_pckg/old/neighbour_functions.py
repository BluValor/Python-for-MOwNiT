from enum import Enum
import numpy as np

# o+o
# +++
# o+o
def no_edges_same_colour_plus(image, coords, y_shape, x_shape):

    sum = 0

    for y in range(-1, 2):
        for x in range(-1, 2):

            ny = coords[0] + y
            if ny >= y_shape:
                ny = ny % y_shape
            nx = coords[1] + x
            if nx >= x_shape:
                nx = nx % x_shape

            if y == 0 or x == 0:
                if image[ny, nx] != image[coords]:
                    sum += 1

            else:
                if image[ny, nx] == image[coords]:
                    sum += 1

    return sum


def no_edges_same_colour_plus_no_corners(image, coords, y_shape, x_shape):

    sum = 0
    color = image[coords]
    y_coord, x_coord = coords

    if y_coord != y_shape - 1:
        if image[y_coord + 1][x_coord] != color:
            sum += 1

    if y_coord != 0:
        if image[y_coord - 1][x_coord] != color:
            sum += 1

    if x_coord != x_shape - 1:
        if image[y_coord][x_coord + 1] != color:
            sum += 1

    if x_coord != 0:
        if image[y_coord][x_coord - 1] != color:
            sum += 1

    return sum


# o+o
# +++
# o+o
def with_edges_same_colour_plus(image, coords, y_shape, x_shape):

    sum = 0

    for y in range(-1, 2):
        for x in range(-1, 2):

            ny = coords[0] + y
            if ny >= y_shape or ny < 0:
                break
            nx = coords[1] + x
            if nx >= x_shape or nx < 0:
                break

            if y == 0 or x == 0:
                if image[ny, nx] != image[coords]:
                    sum += 1

            else:
                if image[ny, nx] == image[coords]:
                    sum += 1

    return sum


# oo+oo
# oo+oo
# +++++
# oo+oo
# oo+oo
def no_edges_same_colour_big_plus(image, coords, y_shape, x_shape):

    sum = 0

    for y in range(-2, 3, 1):
        for x in range(-2, 3, 1):

            ny = coords[0] + y
            if ny >= y_shape:
                ny = ny % y_shape
            nx = coords[1] + x
            if nx >= x_shape:
                nx = nx % x_shape

            if y == 0 or x == 0:
                if image[ny, nx] != image[coords]:
                    sum += 1

            else:
                if image[ny, nx] == image[coords]:
                    sum += 1

    return sum


# +++++
# +ooo+
# +o+o+
# +ooo+
# +++++
def no_edges_same_colour_big_square(image, coords, y_shape, x_shape):

    sum = 0

    for y in range(-2, 3, 1):
        for x in range(-2, 3, 1):

            ny = coords[0] + y
            if ny >= y_shape:
                ny = ny % y_shape
            nx = coords[1] + x
            if nx >= x_shape:
                nx = nx % x_shape

            if (y % 2 == 0 and y != 0) or (x % 2 == 0 and x != 0):
                if image[ny, nx] != image[coords]:
                    sum += 1

            else:
                if image[ny, nx] == image[coords]:
                    sum += 1

    return sum