import pandas as pd
import sys
from matplotlib import pyplot as plt

if __name__ == "__main__":
    file_addr = sys.argv[1]
    d = pd.read_csv(file_addr)
    d.loss.plot(label='Total')
    d.loss.rolling(50).mean().plot(label="Mean (50 ep)")
    #cum_reward = d.groupby('episode').sum()['reward']
    #cum_reward.plot(label='Total')
    #cum_reward.rolling(50).mean().plot(label='Mean (50 ep)')
    plt.grid()
    plt.ylabel('Loss')
    plt.xlabel('Step')
    plt.legend()
    plt.show()
