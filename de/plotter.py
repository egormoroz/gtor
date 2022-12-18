import matplotlib.pyplot as plt
import numpy as np

data = []
with open('100_128.txt') as f:
    for i in f:
        data.append(list(map(float, i.strip().split())))

m = len(min(data, key=len))
n = len(data)

A = np.zeros((n, m))
for i, y in enumerate(data):
    A[i] = y[:m]


plt.ylim(top=1000)
x = np.arange(m)
for i in range(n):
    plt.plot(np.arange(m), A[i], 'g')

mean = np.mean(A, axis=0)
sigma = np.std(A, axis=0)

plt.plot(x, mean, 'r',
         x, mean + 3*sigma, 'b--', 
         x, mean - 3*sigma, 'b--',
         x, np.zeros_like(x), '--')
plt.show()
