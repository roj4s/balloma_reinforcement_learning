import cv2
import os
import numpy as np
import math

class DigitsMatcher:

    def __init__(self, digits_folder_addr):
        self.digits_folder_addr = digits_folder_addr
        self.digits_imgs = [cv2.imread(os.path.join(self.digits_folder_addr,
                                               f"{i}.png")) for
                       i in range(8)]
        self.digits_bw = [self.preprocess_bw(img) for img in self.digits_imgs]

    def preprocess_bw(self, cropped_frame):
        cut = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
        _, cut = cv2.threshold(cut, 200, 255, cv2.THRESH_BINARY)
        return cut

    def match(self, cropped_frame):
        cropped_bw = self.preprocess_bw(cropped_frame)
        cropped_bw_shape = cropped_bw.shape
        d = None
        m = math.inf
        for i, dig in enumerate(self.digits_bw):
            dig = cv2.resize(dig, (cropped_bw_shape[1], cropped_bw_shape[0]))
            dif = np.array(cropped_bw - dig).flatten()
            s = sum(dif)
            if s < m:
                m = s
                d = i

        return d

if __name__ == "__main__":
    from asb import AndroidScreenBuffer
    import cv2

    asb = AndroidScreenBuffer(1313,
                              scale_ratio=0.1,
                              bitrate=120000
                              )

    asb.run()
    dm = DigitsMatcher("/home/neo/dev/balloma_rl_agent/misc/digits")

    output_addr = "/home/neo/dev/balloma_rl_agent/misc/digits"
    i = 0
    while True:
        img = asb.get_last_frame()
        if img is not None:
            cv2.imshow('capture', img)
            frm = img[11:27, 25:35]
            cv2.imshow('diamonds_gathered', frm)
            dig = dm.match(frm)
            print(dig)

            '''
            cv2.imshow('diamonds_total', img[11:26, 48:58])
            #cv2.imshow('time start', img[117:132, 30:41])
            j = 0
            for i in range(31, 31+9*2, 9):
                j += 1
                frm = img[117:132,i:i+9]
                dig = dm.match(frm)
                #print(dig)
                try:
                    frm = cv2.putText(frm, str(dig), (0, 0), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
                except Exception as e:
                    print(e)
                    pass
                cv2.imshow(f'time_entry_{j}', frm)
            '''


        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            asb.stop()
            exit(0)
            break
