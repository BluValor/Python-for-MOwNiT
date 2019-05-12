import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
import random


def apply_transform_matrix(X, Y, Z, A, size):

    for i in range(size):
        for j in range(size):
            pre = np.row_stack([X[i][j], Y[i][j], Z[i][j]])
            after = A @ pre
            X[i][j] = after[0][0]
            Y[i][j] = after[1][0]
            Z[i][j] = after[2][0]


def draw_semi_major_axes(U, S):

    center = [0, 0, 0]
    x_axis = [1, 0, 0]
    y_axis = [0, 1, 0]
    z_axis = [0, 0, 1]

    x_transformed = U @ S @ x_axis
    y_transformed = U @ S @ y_axis
    z_transformed = U @ S @ z_axis

    x_line = [list(t) for t in zip(center, x_transformed.tolist()[0])]
    y_line = [list(t) for t in zip(center, y_transformed.tolist()[0])]
    z_line = [list(t) for t in zip(center, z_transformed.tolist()[0])]

    ax.plot(x_line[0], x_line[1], x_line[2], color='red')
    ax.plot(y_line[0], y_line[1], y_line[2], color='green')
    ax.plot(z_line[0], z_line[1], z_line[2], color='blue')


def random_elipse(X, Y, Z, size):

    A = np.random.rand(3, 3)
    U, S_list, VH = np.linalg.svd(A)
    S = np.diag(S_list)
    U, S, VH = np.matrix(U), np.matrix(S), np.matrix(VH)

    print("U:\n", U, '\n\nS:\n', S, '\n\nVH:\n', VH, "\n\nA:\n", A)

    apply_transform_matrix(X, Y, Z, A, size)

    draw_semi_major_axes(U, S)


def ex4(X, Y, Z, size):

    U = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    S = np.matrix([[100, 0, 0], [0, 10, 0], [0, 0, 0.5]])
    VH = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    A = U @ S @ VH

    print("U:\n", U, '\n\nS:\n', S, '\n\nVH:\n', VH, "\n\nA:\n", A)

    apply_transform_matrix(X, Y, Z, A, size)

    draw_semi_major_axes(U, S)
    ax.plot([-100, 100], [-100, 100], [-100, 100], color='white')


def ex5(X, Y, Z, size):

    A = [[0.24895347, 0.43064318, 0.31522827],
         [0.32584478, 0.03375932, 0.90251862],
         [0.47745742, 0.31790023, 0.10860795]]

    U, S_list, VH = np.linalg.svd(A)
    S = np.diag(S_list)
    U, S, VH = np.matrix(U), np.matrix(S), np.matrix(VH)

    print("U:\n", U, '\n\nS:\n', S, '\n\nVH:\n', VH, "\n\nA:\n", A)

    def ex5_VH():
        for i in range(size):
            for j in range(size):
                pre = np.row_stack([X[i][j], Y[i][j], Z[i][j]])
                after = VH @ pre
                X[i][j] = after[0][0]
                Y[i][j] = after[1][0]
                Z[i][j] = after[2][0]

        draw_semi_major_axes(np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))

    def ex5_SVH():
        for i in range(size):
            for j in range(size):
                pre = np.row_stack([X[i][j], Y[i][j], Z[i][j]])
                after = S @ VH @ pre
                X[i][j] = after[0][0]
                Y[i][j] = after[1][0]
                Z[i][j] = after[2][0]

        draw_semi_major_axes(np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), S)

    def ex5_USVH():
        for i in range(size):
            for j in range(size):
                pre = np.row_stack([X[i][j], Y[i][j], Z[i][j]])
                after = U @ S @ VH @ pre
                X[i][j] = after[0][0]
                Y[i][j] = after[1][0]
                Z[i][j] = after[2][0]

        draw_semi_major_axes(U, S)

    ax.plot([-1, 1], [-1, 1], [-1, 1], color='white')
    # ex5_VH()
    # ex5_SVH()
    ex5_USVH()


size = 20

arg1, arg2 = np.linspace(0, 2 * np.pi, size), np.linspace(0, np.pi, size)
mesh1, mesh2 = np.meshgrid(arg1, arg2)

R = np.cos(mesh2 ** 2)
X = np.cos(mesh1) * np.sin(mesh2)
Y = np.sin(mesh1) * np.sin(mesh2)
Z = np.cos(mesh2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

random_elipse(X, Y, Z, size)
# ex4(X, Y, Z, size)
# ex5(X, Y, Z, size)

plot = ax.plot_surface(
    X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=False, alpha=0.2)  # cmap=plt.get_cmap('jet')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()
