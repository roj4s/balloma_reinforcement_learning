from matplotlib import pyplot as plt
import sys
import pandas as pd
import numpy as np
import sys

def plot(file_addr):
    v = sys.argv[2:]
    fig, axes = plt.subplots(len(v))
    if len(v) == 1:
        axes = [axes]

    while True:
        d = pd.read_csv(file_addr)
        #d = d[d.episode < 120]
        episodes = d.episode.unique()
        x = [i for i in episodes]

        for i, vv in enumerate(v):
            val = d.groupby('episode').mean()[vv]
            axes[i].plot(x, val, label="Total")
            rolling = val.rolling(50).mean()
            axes[i].plot(x, rolling, label="Mean (50 ep)")
            axes[i].set_ylabel(vv)
            axes[i].grid()

        axes[-1].set_xlabel('Episode')
        plt.legend(prop={'size': 8})
        plt.pause(0.5)


        for i, vv in enumerate(v):
            axes[i].cla()

if __name__ == "__main__":
    plot(sys.argv[1])
