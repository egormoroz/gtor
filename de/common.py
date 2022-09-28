import numpy as np

F = 0.8
CR = 0.9

NP = 100
NDIM = 100
EPOCHS = 100000

EPS = 1e-8

FILE_PATH = 'data.npy'

def fitness(X):
    A = 10
    return A * NDIM + X @ X - A * np.cos(2 * np.pi * X).sum()

def mean_fitness(pops):
    fs = [fitness(pops[i]) for i in range(NP)]
    return sum(fs) / NP

def best_fitness(pops):
    fs = [fitness(pops[i]) for i in range(NP)]
    idx = np.argmin(fs)
    return idx, fs[idx]


