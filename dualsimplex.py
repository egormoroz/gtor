import numpy as np

EQ = 0
LE = 1
GE = -1

def construct_table(A, b, c, s):
    m, n = A.shape
    assert len(b) == m
    assert len(s) == m
    assert len(c) == n

    E = np.zeros((m, m - s.count(EQ)))
    R = np.zeros((m, m - s.count(LE)))

    j, k = 0, 0
    for i, sgn in enumerate(s):
        if sgn != EQ:
            E[i, j] = sgn
            j += 1

        if sgn != LE:
            R[i, k] = 1
            k += 1

    D = np.hstack((A, E, R, b[:, np.newaxis]))
    T = np.zeros(n + E.shape[1] + R.shape[1] + 1)
    q = R.shape[1] + 1 
    for i, sgn in enumerate(s):
        if sgn != LE:
            T[:-q] += D[i, :-q]
            T[-1] += D[i, -1]

    basis = np.zeros(m, dtype=np.int64)
    p, q = 0, 0
    for i, sgn in enumerate(s):
        if sgn == LE:
            basis[i] = n + p
        else:
            basis[i] = n + E.shape[1] + q

        if sgn != EQ:
            p += 1
        else:
            q += 1

    return T, D, np.array(basis)


def simplex(D, T, basis):
    while True:
        pc = np.argmax(T[:-1])
        if T[pc] < 1e-8:
            break

        mask = np.where(D[:, pc] > 1e-8)[0]
        pr = mask[np.argmin(D[mask, -1] / D[mask, pc])]

        basis[pr] = pc

        D[pr] /= D[pr, pc]
        T -= T[pc] * D[pr]

        for i in range(D.shape[0]):
            if i != pr:
                D[i] -= D[i, pc] * D[pr]

    x = np.zeros(len(T) - 1)
    np.put(x, basis, D[:, -1])
    return (x, T[-1])


def dual_simplex(A, b, c, s):
    T, D, basis = construct_table(A, b, c, s)
    x, z = simplex(D, T, basis)
    print(np.vstack((T, D)))

    n_r = len(s) - s.count(LE)
    D = np.hstack((D[:, :-(n_r + 1)], D[:,-1,np.newaxis]))
    T = np.hstack((-c, np.zeros(D.shape[1] - len(c))))

    for i, j in enumerate(basis):
        T -= T[j] * D[i]

    x, z = simplex(D, T, basis)
    return x, z

    
# A = np.array([[3, 1], [4, 3], [1, 2]])
# b = np.array([3, 6, 4])
# c = np.array([4, 1])
# s = [EQ, GE, LE]

A = np.array([[1, 1], [2, 1], [1, 2], [1, 1]])
b = np.array([3, 5, 5, 1])
c = -np.array([2, 3])
s = [LE, LE, LE, GE]

print(dual_simplex(A, b, c, s))

