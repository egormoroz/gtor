import numpy as np
from pysmps import smps_loader as smps

'''
Minimize <c, x>, s.t. Ax <= b

Shapes:
    c: (m, 1)
    x: (m, 1)
    A: (n, m)
    b: (n, 1)
'''
def solve(c, A, b):
    EPS = 1e-12
    assert np.all(b >= 0)

    n, m = A.shape

    basis = np.arange(start=m, stop=n+m, step=1, dtype=int)
    D = np.hstack((A, np.eye(n), b))
    T = np.hstack((-c.T, np.zeros((1, n + 1))))

    pivot_col = np.argmax(T)
    pivot_val = T[0, pivot_col]

    while pivot_val >= 0.5:
        mask = np.where(D[:, pivot_col] > EPS)[0]
        pivot_row = mask[(D[mask, -1] / D[mask, pivot_col]).argmin()]
        basis[pivot_row] = pivot_col

        D[pivot_row, :] /= D[pivot_row, pivot_col]
        T -= T[0, pivot_col] * D[pivot_row, :]

        for i in range(n):
            if i == pivot_row:
                continue
            D[i, :] -= D[i, pivot_col] * D[pivot_row, :]

        pivot_col = np.argmax(T)
        pivot_val = T[0, pivot_col]
        print(T[0, -1], pivot_val)

    x = np.zeros(n+m)
    np.put(x, basis, D[:, -1])
    return (x[:m].reshape((m, 1)), T[0, -1])

# c = np.array([[2, 3]]).T
# A = np.array([[1, 1], [6, 3], [1, 2]])
# b = np.array([[100, 360, 120]]).T
#The solution is (40, 40)

# c = np.array([[1, 2, -1]]).T
# A = np.array([[2, 1, 1], [4, 2, 3], [2, 5, 5]])
# b = np.array([[14, 28, 30]]).T
#The solution is (5, 4, 0)

# print(solve(c, A, b))

#mps = smps.load_mps('gen-ip002.mps')
mps = smps.load_mps('reblock115.mps')

c = mps[6]
A = mps[7]
b = mps[9]['rhs']

c = c.reshape((c.shape[0], 1))
b = b.reshape((b.shape[0], 1))

x, cx = solve(c, A, b)

print(x)

print(f'{cx} vs {c.T @ x}')
print(np.max(A @ x - b))

