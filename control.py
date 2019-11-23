import subprocess as sp
from subprocess import PIPE
import numpy as np
import time
import random
from android_screen import AndroidScreenBuffer
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

if __name__ == "__main__":
    asb = AndroidScreenBuffer()
    asb.run()
    coords = [45, 60, 118, 180]
    while True:
        img = asb.get_last_frame()
        if img is not None:
            cv2.imshow('capture', img)
            og = on_game(img, coords, 's8_cut_try_again.png')
            print("On game: {}".format(og))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            asb.stop()
            exit(0)
            break


