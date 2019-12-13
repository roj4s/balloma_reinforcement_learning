from matplotlib import pyplot as plt
import sys
import pandas as pd
import numpy as np

def plot(file_addr):
    fig, axes = plt.subplots(2)
    
    while True:
        d = pd.read_csv(file_addr)
        axes[0].set_ylabel('Reward')
        axes[1].set_ylabel('Loss')
        axes[0].grid()
        axes[1].grid()
        x = [i for i in range(d.shape[0])]
        axes[0].plot(x, d.reward)
        reward_rolling = d.reward.rolling(100).mean()
        axes[0].plot(np.arange(reward_rolling.shape[0]), reward_rolling)
        axes[1].plot(x, d.loss)
        loss_rolling = d.loss.rolling(100).mean()
        axes[1].plot(np.arange(loss_rolling.shape[0]), loss_rolling)
        plt.pause(0.5)
        axes[0].cla()
        axes[1].cla()

if __name__ == "__main__":
    plot(sys.argv[1])
