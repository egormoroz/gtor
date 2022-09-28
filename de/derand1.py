import numpy as np

from common import *

def search_iteration(pops):
    pops_old = np.array(pops, copy=True)
    _, best_val = best_fitness(pops)

    for i in range(NP):
        x = pops_old[i]

        indices = np.random.choice(NP, size=4, replace=False)
        a, b, c = pops_old[indices[indices != i]][:3]

        R = np.random.randint(NDIM)
        r = np.random.random(NDIM)

        y = np.array(x, copy=True)
        v = a + F * (b - c)
        y[r < CR] = v[r < CR]
        y[R] = a[R] + F * (b[R] - c[R])

        fx, fy = fitness(x), fitness(y)
        if fy <= fx:
            pops[i] = y


