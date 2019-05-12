import numpy as np

np.set_printoptions(precision=3, suppress=True)


def gj_column_diagonal(mA, column, size):

    for i in range(column + 1, size):
        factor = (-1) * mA[i][column] / mA[column][column]
        mA[i] += factor * mA[column]



def ex1_diagonal_pivot(m_size, A, B):

    X = np.c_[A, B]

    for column in range(m_size):
        gj_column_diagonal(X, column, m_size)

    for x in range(m_size):
        for j in range(m_size):
            if abs(X[x][j]) < 1e-8:
                X[x][j] = 0

    A = X[:, :-1]
    B2 = np.zeros(m_size)
    for i in range(m_size):
        B2[i] = X[i][-1]

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

    # print("A * X = B")
    # print("\nA:")
    # print(np.matrix(A))
    # print("\nB:")
    # print(B2)
    # print("Result + number of variable:")
    # print(result)
    # variables = ["x" + str(i) for i in range(0, m_size)]
    # print(variables)
