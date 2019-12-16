from matplotlib import pyplot as plt
import sys
import pandas as pd
import numpy as np

def plot(file_addr):
    fig, axes = plt.subplots(5)

    while True:
        d = pd.read_csv(file_addr)
        x = [i for i in range(d.shape[0])]
        v = ['reward', 'loss', 'vector_size', 'angle', 'speed']

        for i, vv in enumerate(v):
            axes[i].plot(x, d[vv])
            rolling = d[vv].rolling(100).mean()
            axes[i].plot(np.arange(rolling.shape[0]), rolling)
            axes[i].set_ylabel(vv)
            axes[i].grid()


        plt.pause(0.5)


        for i, vv in enumerate(v):
            axes[i].cla()

if __name__ == "__main__":
    plot(sys.argv[1])
