import subprocess as sp
from subprocess import PIPE
import numpy as np
import time
import random
import cv2

def put(size, angle, duration, device_width=1440, device_height=2960):
    x0 = int(device_height*0.9)
    y0 = int(device_width*0.5)
    x1 = x0 + int((size * np.sin(90 - angle)))
    y1 = y0 - int((size * np.cos(angle)))
    print("Swapping from {} to {}, angle: {}, in {} secs".format(str((x0, y0)), str((x1,
                                                                         y1)),
                                                                 angle,
          duration))
    sp.call(["adb", "shell", "input", "touchscreen", "swipe", str(x0),
                     str(y0),
                     str(x1), str(y1), str(duration), ";"])

def tap(x, y):
    sp.call(['adb', 'shell', 'input', 'tap', str(x), str(y)])

def on_game(frame, el_coord, el_img, threshold=4):
    '''
        Currently checking by comparing images.
        It will be replaced by a less device-dependent approach.
    '''
    cut = frame[el_coord[0]:el_coord[1], el_coord[2]:el_coord[3]]
    cut = cv2.cvtColor(cut, cv2.COLOR_BGR2GRAY)
    _, cut = cv2.threshold(cut, 200, 255, cv2.THRESH_BINARY)
    #cv2.imshow('cut', cut)
    el = cv2.imread(el_img)
    el = cv2.cvtColor(el, cv2.COLOR_BGR2GRAY)
    _, el = cv2.threshold(el, 200, 255, cv2.THRESH_BINARY)
    #cv2.imshow('el', el)
    dif = np.array(cut - el).flatten()
    return sum(dif) > threshold

def transform_action(action, action_range, action_low):
    return action * action_range + action_low

if __name__ == "__main__":
    from ddpg import OUNoise
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

    fig, ax = plt.subplots(3, sharex='col', sharey='row')
    for i in range(3):
        ax[i].plot(iis, values[i])

    plt.show()

        #put(*action, device_width=w, device_height=h)




