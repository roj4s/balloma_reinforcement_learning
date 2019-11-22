import subprocess as sp
from subprocess import PIPE
import numpy as np
import time
import random

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



if __name__ == "__main__":
    wss = sp.check_output(['adb', 'shell', 'wm', 'size'])
    w = str(wss).split('x')[0]
    w = int(w[w.index(":") + 1:])
    h = str(wss).split('x')[1]
    h = int(h[:h.index("\\n")])

    for i in range(20):
        size = random.randint(30, 100)
        angle = random.randint(0, 180)
        put(size, angle, random.randint(100, 1000), w, h)
        time.sleep(0.5)
