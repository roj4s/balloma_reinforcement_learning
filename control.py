import subprocess
import numpy as np
import time
import random

def put(size, angle, duration):
    x0 = 1326
    y0 = 459
    x1 = x0 + int((size * np.sin(90 - angle)))
    y1 = y0 + int((size * np.cos(angle)))
    print("Swapping from {} to {} in {} secs".format(str((x0, y0)), str((x1,
                                                                         y1)),
          duration))
    subprocess.call(["adb", "shell", "input", "touchscreen", "swipe", str(x0),
                     str(y0),
                     str(x1), str(y1), str(duration), ";"])


for i in range(20):
    put(random.randint(30, 100), random.randint(0, 180), random.randint(100, 1000))
    time.sleep(0.5)
