if __name__ == "__main__":
    from ounoise import OUNoise
    from asb import AndroidScreenBuffer
    from matplotlib import pyplot as plt
    #buff = AndroidScreenBuffer()
    #h, w = buff.get_device_screen_shape()
    exploration_mu = 0
    exploration_theta = 0.15
    exploration_sigma = 0.2
    action_size = 3
    action_low = np.array([1, 0, 1])
    action_high = np.array([10, 359, 2000])
    action_range = action_high - action_low

    action = np.array([np.random.uniform() for _ in action_low])
    noise = OUNoise(action.shape[0], exploration_mu,
                                 exploration_theta, exploration_sigma)
    values = np.zeros((100, 3))
    iis = [i for i in range(100)]
    for i in iis:
        action = action + noise.sample()
        action = np.array(transform_action(action, action_range, action_low),
                          dtype='uint8')
        values[i] = action

    print(values.shape)

    fig, ax = plt.subplots(3, sharex='col', sharey='row')
    labels = ['Size', 'Angle', 'Duration']
    for i in range(3):
        ax[i].plot(iis, values[:,i])
        ax[i].set_xlabel(labels[i])

    plt.grid()
    plt.show()

    #put(*action, device_width=w, device_height=h)
