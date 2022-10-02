import numpy as np
from pysmps import smps_loader as smps
import time

def simplex(D, T, n, m):
    basis = np.arange(start=m, stop=n+m, step=1, dtype=int)

    k = 0
    while True:
        #eblan?
        # x = np.zeros(n+m)
        # np.put(x, basis, D[:, -1])
        # print(k, T[-1] - c @ x[:m], np.max(A @ x[:m] - b),
        #         np.max((D[:,:-1] @ x) - D[:,-1]), T[-1])

        pc = np.argmax(T[:-1])
        if T[pc] < 0.5:
            break

        mask = np.where(D[:, pc] > 1e-8)[0]
        pr = mask[np.argmin(D[mask, -1] / D[mask, pc])]

        basis[pr] = pc

        D[pr] /= D[pr, pc]
        T -= T[pc] * D[pr]

        for i in range(D.shape[0]):
            if i != pr:
                D[i] -= D[i, pc] * D[pr]

        k += 1
        print('{:04d} {}'.format(k, T[-1]))


    x = np.zeros(n+m)
    np.put(x, basis, D[:, -1])
    print(T[-1] - c @ x[:m], np.max(A @ x[:m] - b),
            np.max((D[:,:-1] @ x) - D[:,-1]), T[-1])

    return (x[:m], T[-1])


mps = smps.load_mps('reblock115.mps')

c = mps[6]
A = mps[7]
b = mps[9]['rhs']

n, m = A.shape

D = np.hstack((A, np.eye(n), b[:, np.newaxis]))
T = np.hstack((-c, np.zeros(n + 1)))



start = time.time()
x, z = simplex(D, T, n, m)
end = time.time()
print('elapsed', end - start)

