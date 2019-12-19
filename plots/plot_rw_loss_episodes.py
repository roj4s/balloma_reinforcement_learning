from matplotlib import pyplot as plt
import sys
import pandas as pd
import numpy as np

def plot(file_addr):
    #v = ['reward', 'loss']
    v = ['reward']
    fig, axes = plt.subplots(len(v))
    if len(v) == 1:
        axes = [axes]

    while True:
        d = pd.read_csv(file_addr)
        d = d[d.episode < 120]
        episodes = d.episode.unique()
        x = [i for i in episodes]
        print(min(x))
        print(max(x))

        for i, vv in enumerate(v):
            val = d.groupby('episode').mean()[vv]
            axes[i].plot(x, val)
            rolling = d[vv].rolling(50).mean()
            axes[i].plot(x, rolling)
            axes[i].set_ylabel(vv)
            axes[i].grid()

        axes[-1].set_xlabel('Episode')
        plt.pause(0.5)


        for i, vv in enumerate(v):
            axes[i].cla()

if __name__ == "__main__":
    plot(sys.argv[1])
