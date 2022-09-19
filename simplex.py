import numpy as np

'''
Minimize <c, x>, s.t. Ax <= b

Shapes:
    c: (m, 1)
    x: (m, 1)
    A: (n, m)
    b: (n, 1)
'''
def solve(c, A, b):
    assert np.all(b >= 0)

    n, m = A.shape

    basis = np.arange(start=m, stop=n+m, step=1, dtype=int)
    D = np.hstack((A, np.eye(n), b))
    T = np.hstack((-c.T, np.zeros((1, n + 1))))

    pivot_col = np.argmin(T)
    pivot_val = T[0, pivot_col]

    while pivot_val < -1e-8:
        pivot_row = np.argmin(D[:, -1] / D[:, pivot_col])
        basis[pivot_row] = pivot_col

        D[pivot_row, :] /= D[pivot_row, pivot_col]
        T -= T[0, pivot_col] * D[pivot_row, :]

        for i in range(n):
            if i == pivot_row:
                continue
            D[i, :] -= D[i, pivot_col] * D[pivot_row, :]

        pivot_col = np.argmin(T)
        pivot_val = T[0, pivot_col]

    x = np.zeros(n+m)
    np.put(x, basis, D[:, -1])
    return (x[:m], T[0, -1])

c = np.array([[2, 3]]).T
A = np.array([[1, 1], [6, 3], [1, 2]])
b = np.array([[100, 360, 120]]).T
#The solution is (40, 40)

# c = np.array([[1, 2, -1]]).T
# A = np.array([[2, 1, 1], [4, 2, 3], [2, 5, 5]])
# b = np.array([[14, 28, 30]]).T
#The solution is (5, 4, 0)

print(solve(c, A, b))

