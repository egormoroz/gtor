import numpy as np
import os

from common import *


class Archive:
    def __init__(self):
        self.data = np.zeros((NP, NDIM))
        self.size = 0

    def add_entry(self, entry):
        self.size = min(self.size + 1, NP)
        self.data[np.random.randint(self.size)] = entry

    def get_entry(self):
        return self.data[np.random.randint(self.size)]


np.random.seed(1)
pops = np.random.randn(NP, NDIM)

if os.path.isfile(FILE_PATH) and False:
    print('loaded population data from file ', FILE_PATH)
    pops = np.load(FILE_PATH)


_, best_val = best_fitness(pops)
archive = Archive()

nu_CR = nu_F = 0.5
c = 0.1
p_NP = max(1, int(0.05 * NP))


for epoch in range(EPOCHS):
    pops_old = np.array(pops, copy=True)

    fs = np.array([fitness(pops[i]) for i in range(NP)])
    sorted_ics = np.argsort(fs)

    cr = nu_CR + 0.1 * np.random.standard_normal(size=NP)
    f = nu_F + 0.1 * np.random.standard_cauchy(size=NP)

    cr, f = np.clip(cr, 0, 1), np.clip(f, 0, 1)
    r = np.random.random((NP, NDIM))

    S_CR, S_F = np.zeros(NP), np.zeros(NP)
    n_successes = 0

    for i in range(NP):
        x = pops_old[i]
        pbest_idx = sorted_ics[np.random.randint(p_NP)]
        pbest = pops_old[pbest_idx]

        ics = np.random.choice(NP, size=4, replace=False)
        ics = ics[(ics != pbest_idx) & (ics != i)][:2]
        
        x1, x2 = pops_old[ics]
        if archive.size > 0 and np.random.randint(2) > 0:
            x2 = archive.get_entry()

        R = np.random.randint(NDIM)
        r = np.random.random(NDIM)

        y = x + f[i] * (pbest - x) + f[i] * (x1 - x2)
        y_R = y[R]
        y[r >= cr[i]] = x[r >= cr[i]]
        y[R] = y_R

        fx, fy = fitness(x), fitness(y)
        if fy <= fx:
            pops[i] = y
            archive.add_entry(x)

            S_CR[n_successes] = cr[i]
            S_F[n_successes] = f[i]
            n_successes += 1
            

    mean_A = np.sum(S_CR) / n_successes if n_successes != 0 else 0
    mean_L = (S_F @ S_F) / np.sum(S_F) if np.any(S_F.nonzero()) else 0

    nu_CR = (1 - c) * nu_CR + c * mean_A
    nu_F = (1 - c) * nu_F + c * mean_L

    _, f = best_fitness(pops)
    print('{:03d}. mean: {:.4f} best: {:.4f}'.format(
            epoch, mean_fitness(pops), f), 
        end='\n' if f < best_val else '\r')
    if f < best_val:
        best_val = f
        # np.save(FILE_PATH, pops)

    if best_val < EPS:
        break

idx, val = best_fitness(pops)
print()
# print('x_min: ', pops[idx])
print('fitness(x_min) =', val)


