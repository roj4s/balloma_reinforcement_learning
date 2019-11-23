import numpy as np
from control import put, on_game
import time
from threading import Event
from queue import Queue
from android_screen import AndroidScreenBuffer
import cv2


class Environment:
    """Environment, defines the goal and provides feedback to the agent."""
    def __init__(self, width=240, height=240, minicap_port=1313,
                 device_ref_elements_data={}):
        self.asb = AndroidScreenBuffer(minicap_port=minicap_port)
        self.device_height, self.device_width = self.asb.get_device_screen_shape()
        self.action_repeat = 3
        self.state_frame_width = width
        self.state_frame_height = height
        self.device_evt = Event()
        self.state_size = self.action_repeat * self.state_frame_width * self.state_frame_height * 3
        self.device_ref_elements_data = device_ref_elements_data
        self.asb.run()

    def get_reward(self, state_frame):
        """
            Given a state frame extracts reward
            involved view components, converts it to
            float representation and computes reward.
        """
        # TODO:
        return np.random.random()

    def step(self, vector_size, angle, speed, show_frames=False):
        """
            Uses action to obtain next state, reward, done.
            Through adb shell input applies a touchscreen swipe
            with specified vector size, angle and velocity, captures
            a game frame after applying such action to determine
            reward, next state and check if scene was completed.
        """
        # TODO:
        reward = 0
        states = []
        for _ in range(self.action_repeat):
            put(vector_size, angle, speed, self.device_width, self.device_height)
            t = time.time()
            frame = None
            while frame is None:
                #frame = self.asb.get_timelapse_frame(t)
                frame = self.asb.get_last_frame()
            if show_frames and frame is not None:
                cv2.imshow('frame', frame)
            reward += self.get_reward(frame)
            states.append(reward)
            done = self.is_done(frame)
        #next_state = np.concatenate(states)
        next_state = []
        return next_state, reward, done

    def is_done(self, frame):
        '''
            Currently checking by comparing images.
            It will be replaced by a less device-dependent approach.
        '''
        try_again_el_data = self.device_ref_elements_data['try_again']
        coords = try_again_el_data['coords']
        img_path = try_again_el_data['img_path']
        return not on_game(frame, coords, img_path)

    def reset(self):
        """
            Put the environment back on game
        """
        # TODO
        return True

if __name__ == "__main__":
    import random
    try_again_el_data = {
        'coords': [45, 60, 118, 180],
        'img_path': 'data/s8_cut_try_again.png'
    }
    env = Environment(device_ref_elements_data={'try_again': try_again_el_data})

    done = False
    while not done:
        angle = random.randint(0, 359)
        v_size = random.randint(1, 50)
        speed = random.randint(100, 2000)
        ns, rw, done = env.step(v_size, angle, speed, show_frames=False)

    cv2.destroyAllWindows()

