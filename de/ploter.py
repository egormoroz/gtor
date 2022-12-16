import matplotlib.pyplot as plt
import numpy as np

data = []
with open('200_64.txt') as f:
    for i in f:
        data.append(list(map(float, i.strip().split())))

m = len(min(data, key=len))
n = len(data)

A = np.zeros((n, m))
for i, y in enumerate(data):
    A[i] = y[:m]

mean = np.mean(A, axis=0)
mean_diff = np.mean(np.abs(A[1:] - A[:-1]), axis=0)

x = np.arange(m)
plt.ylim(top=1000)
plt.plot(x, mean, 
         x, mean + mean_diff, '--', 
         x, mean - mean_diff, '--')
plt.show()
