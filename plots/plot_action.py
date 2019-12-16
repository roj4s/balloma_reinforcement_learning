import pandas as pd
import sys
from matplotlib import pyplot as plt
import numpy as np



def transform_action(action, action_range, action_low):
    return action * action_range + action_low


if __name__ == "__main__":
    action_low = np.array([1, 0, 1])
    action_high = np.array([10, 359, 2000])
    action_range = action_high - action_low
    file_addr = sys.argv[1]
    d = pd.read_csv(file_addr)
    v = ['vector_size', 'angle', 'speed']
    data = np.array([transform_action(action, action_range, action_low) for
                     action in d.groupby('episode').mean()[v].values])
    x = [i for i in range(data.shape[0])]
    fig, axes = plt.subplots(3)
    for i, vv in enumerate(v):
        axes[i].plot(x, data[:, i])
        axes[i].grid()
        axes[i].set_ylabel(vv)

    axes[2].set_xlabel('Episode')
    plt.show()
