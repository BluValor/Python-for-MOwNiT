import numpy as np

np.set_printoptions(precision=3, suppress=True)


def gj_column_full(mA, column, size, changes):
    r_max = column
    c_max = column

    for i in range(column, size):
        for j in range(column, size):
            if abs(mA[i][j]) > abs(mA[r_max][c_max]):
                r_max = i
                c_max = j

    mA[[r_max, column]] = mA[[column, r_max]]
    mA[:, [c_max, column]] = mA[:, [column, c_max]]
    changes[[c_max, column]] = changes[[column, c_max]]

    for i in range(column + 1, size):
        factor = (-1) * mA[i][column] / mA[column][column]
        mA[i] += factor * mA[column]


def ex1_full_pivot(m_size, A, B):

    # print(np.matrix(A))
    # print(np.matrix(B))

    changes = np.zeros(m_size)
    for i in range(m_size):
        changes[i] = float(i)

    X = np.c_[A, B]

    for column in range(m_size):
        gj_column_full(X, column, m_size, changes)

    for x in range(m_size):
        for j in range(m_size):
            if abs(X[x][j]) < 1e-8:
                X[x][j] = 0

    A = X[:, :-1]
    B2 = np.zeros(m_size)
    for i in range(m_size):
        B2[i] = X[i][-1]

    # print("A * X = B")
    # print("\nA:")
    # print(np.matrix(A))
    # print("\nB:")
    # print(B2)

    def gj_solve(mA, mB, size):

        results = np.zeros(size)

        for i in range(size - 1, -1, -1):

            curr_val = mB[i]
            for j in range(size - 1, i, -1):
                curr_val -= mA[i][j] * results[j]

            results[i] = curr_val / mA[i][i]

        return results

    result = gj_solve(A, B2, m_size)

    return result

    # print("Result + number of variable:")
    # print(result)
    # variables = ["x" + str(int(i)) for i in changes]
    # print(variables)

