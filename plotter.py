import matplotlib.pyplot as plt
import numpy as np

def plot_simplex_output(filename, p):
    y = []
    try:
        with open(filename, mode='r') as f:
            for line in f:
                y.append(float(line.strip().split()[0]))
    except:
        pass
    
    p.set_title(filename)
    p.plot(y)


def plot_de_output(filename, p):
    gens, means, bests = [], [], []
    with open(filename, mode='r') as f:
        for line in f:
            _, s = line.strip().split(']')
            w = s.strip().split()
            gens.append(int(w[0]))
            means.append(float(w[2]))
            bests.append(float(w[4]))

    p.plot(gens, bests, 'r', label='best')
    p.plot(gens, means, 'b', label='mean')
    p.set_title(filename)
    p.legend(loc='upper right')

fig, axs = plt.subplots(2, 3)

plot_simplex_output('cod105_output.txt', axs[0, 0])
plot_simplex_output('reblock115_output.txt', axs[1, 0])

plot_de_output('de/1000_4096.txt', axs[0, 1])
plot_de_output('de/1000_2048.txt', axs[1, 1])
plot_de_output('de/1000_1024.txt', axs[0, 2])
plot_de_output('de/1000_0256.txt', axs[1, 2])

plt.show()

