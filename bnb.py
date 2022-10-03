from dualsimplex import *
from collections import deque
import time

'''
- solves integer linear program
- returns first found feasible solution
'''
def branch_and_bound(A, b, c, s):
    m, n = A.shape

    A = np.vstack((A, np.eye(n)))
    b = np.hstack((b, np.zeros(n)))
    s = np.hstack((s, GE * np.ones(n)))

    q = deque()
    q.append((np.zeros(n), GE * np.ones(n)))

    while q:
        b[-n:], s[-n:] = q.popleft()
        sln = dual_simplex(A, b, c, s)
        if sln is None:
            #relaxed LP is infeasible
            continue
        x, z = sln

        non_ints, = np.where(np.abs(x[:n] - np.round(x[:n])) > 1e-8)
        if len(non_ints) == 0:
            print('done')
            return x[:n], z

        for i in non_ints:
            bb, ss = np.copy(b[-n:]), np.copy(s[-n:])
            bb[-i], ss[-i] = np.floor(x[i]), LE
            q.append((bb, ss))
            bb, ss = np.copy(b[-n:]), np.copy(s[-n:])
            bb[-i], ss[-i] = np.ceil(x[i]), GE
            q.append((bb, ss))

    print('failed')
    return x, z


def main():
    A = np.array([[-2, 2], [-8, 10]])
    b = np.array([1, 13])
    c = -np.ones(2)
    s = np.array([GE, LE])


    start = time.time()
    x, z = branch_and_bound(A, b, c, s)
    end = time.time()
    print(x, z)
    print('elapsed', end - start)

if __name__ == '__main__':
    main()

