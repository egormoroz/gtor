import numpy as np
from pysmps import smps_loader

LE = -1
EQ = 0
GE = 1

def read_mps(file):
    m = smps_loader.load_mps(file)
    I = np.array([i == 'integral' for i in m[4]])
    s_dict = { 'E': EQ, 'L': LE, 'G': GE }
    s = np.array([s_dict[i] for i in m[5]])
    c = m[6]
    A = m[7]
    b = next(iter(m[9].values()))

    return A, b, c, s, I


# minimizes
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

    return D, T, basis


def construct_table(A, b, s):
    m, n = A.shape
    assert len(b) == m
    assert len(s) == m

    neg = b < 0
    if np.any(neg):
        A[neg, :] *= -1
        b[neg] *= -1
        s[neg] *= -1

    ics = np.argsort(s)
    A, b, s = A[ics], b[ics], s[ics]

    eqs = np.where(s == EQ)[0]
    les = np.where(s == LE)[0]
    ges = np.where(s == GE)[0]

    # TODO: first phase is not always neccessary
    # test whether np.zeros() is a feasible starting solution

    n_eq, n_le, n_ge = len(eqs), len(les), len(ges)

    # slack variables
    SL = np.zeros((m, n_le))
    SL[les, np.arange(n_le)] = 1

    # excess variables
    EX = np.zeros((m, n_ge))
    EX[ges, np.arange(n_ge)] = -1

    # artificial variables
    n_avs = n_eq + n_ge
    AR = np.zeros((m, n_avs))
    av_rows = np.hstack((eqs, ges))
    AR[av_rows, np.arange(n_avs)] = 1

    A = np.hstack((A, SL, EX, AR))
    c = np.zeros(A.shape[1])
    c[:-n_avs] = np.sum(A[av_rows, :-n_avs], axis=0)

    D = np.hstack((A, b[:, np.newaxis]))
    T = np.hstack((c, np.sum(b[av_rows])))

    basis = np.zeros(m, dtype=np.int64)
    basis[:n_le] = np.arange(n_le) + n
    basis[n_le:] = np.arange(n_avs) + A.shape[1] - n_avs

    return (D, T, basis), n_avs

def first_phase(A, b, c, s):
    tbl, n_avs = construct_table(A, b, s)
    D, T, basis = simplex(*tbl)

    # infeasible
    if T[-1] > 1e-8:
        return

    total_vars = len(T) - 1
    basis_avs = basis[basis >= total_vars - n_avs]

    T[:] = 0
    T[:len(c)] = -c
    for i, j in enumerate(basis):
        T -= T[j] * D[i]

    # no artificial variables in basis
    if len(basis_avs) == 0:
        T, T[-1] = T[:len(T) - n_avs], T[-1]
        D = np.hstack((D[:, :-n_avs-1], D[:, -1, np.newaxis]))
        return D, T, basis

    # a few artificial variables in basis
    nonneg_org_vars = np.where(T[:-1] >= -1e-8)[0]
    vars_remain = np.hstack((nonneg_org_vars, basis_avs, -1))

    remap = np.zeros(len(T) - 1, dtype=np.int64)
    remap[vars_remain] = np.arange(len(vars_remain))

    T = T[vars_remain]
    D = D[:, vars_remain]

    return D, T, remap[basis]


def two_phase_simplex(A, b, c, s):
    m, n = A.shape
    assert len(b) == m
    assert len(s) == m
    assert len(c) == n

    tbl = first_phase(A, b, c, s)
    if tbl is None:
        #print('infeasible')
        return

    D, T, basis = simplex(*tbl)

    x = np.zeros(D.shape[1])
    x[basis] = D[:, -1]

    return x[:len(c)], T[-1]


def main():
    # A = np.array([[3, 1], [4, 3], [1, 2]])
    # b = np.array([3, 6, 4])
    # c = np.array([4, 1])
    # s = [EQ, GE, LE]

    #A = np.array([[1, 1], [2, 1], [1, 2], [1, 1]])
    #b = np.array([3, 5, 5, 1])
    #c = -np.array([2, 3])
    #s = np.array([LE, LE, LE, GE])

    A, b, c, s, _ = read_mps('flugpl.mps')
    print(two_phase_simplex(A, b, c, s))

if __name__ == "__main__":
    A, b, c, s, _ = read_mps('flugpl.relaxed.mps')
    #A = np.array([[3, 1], [4, 3], [1, 2]])
    #b = np.array([3, 6, 4])
    #c = np.array([4, 1])
    #s = np.array([EQ, GE, LE])

    print(two_phase_simplex(A, b, c, s))

