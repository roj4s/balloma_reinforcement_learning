import numpy as np
from control import put, on_game, tap, transform_action
import time
from threading import Event
from queue import Queue
from android_screen import AndroidScreenBuffer
import cv2
from digit_recognition import DigitsMatcher
from ounoise import OUNoise


class Environment:
    """Environment, defines the goal and provides feedback to the agent."""
    def __init__(self, width=240, height=240, minicap_port=1313,
                 device_ref_elements_data={}):
        self.asb = AndroidScreenBuffer(minicap_port=minicap_port,
                                       scale_ratio=0.1, bitrate=120000)
        self.device_height, self.device_width = self.asb.get_device_screen_shape()
        self.action_repeat = 3
        self.state_frame_width = width
        self.state_frame_height = height
        self.device_evt = Event()
        self.state_size = self.action_repeat * self.state_frame_width * self.state_frame_height * 3
        self.device_ref_elements_data = device_ref_elements_data
        self.digits_matcher = DigitsMatcher(self.device_ref_elements_data['scores']['digits_mask_addr'])
        self.asb.run()
        self.episode_start_time = None

    def get_reward(self, state_frame):
        """
            Given a state frame extracts reward
            involved view components, converts it to
            float representation and computes reward.
        """
        # TODO:
        t = time.time() - self.episode_start_time
        match_threshold = self.device_ref_elements_data['scores']['match_threshold']
        episode_time_limit = self.device_ref_elements_data['scores']['episode_time_limit']
        diamonds_total = self.device_ref_elements_data['scores']['diamonds_total']
        diamonds_importance = self.device_ref_elements_data['scores']['diamonds_importance']
        time_importance = self.device_ref_elements_data['scores']['time_importance']
        y1, y2, x1, x2 = self.device_ref_elements_data['scores']['coords_diamonds_gathered']
        cropped_dig = state_frame[y1:y2, x1:x2]
        cv2.imwrite(f'/tmp/frames/dig_{time.time()}.png', np.copy(cropped_dig))
        diamonds_gathered = self.digits_matcher.match(cropped_dig,
                                                      threshold=match_threshold)
        print(f"Diamonds Gathered: {diamonds_gathered}")
        if diamonds_gathered is None:
            #TODO: Enhance this
            return 0

        return diamonds_importance * (diamonds_gathered/diamonds_total) - time_importance * t/episode_time_limit


    def step(self, vector_size, angle, speed):
        """
            Uses action to obtain next state, reward, done.
            Through adb shell input applies a touchscreen swipe
            with specified vector size, angle and velocity, captures
            a game frame after applying such action to determine
            reward, next state and check if scene was completed.
        """
        # TODO:
        if self.episode_start_time is None:
            self.episode_start_time = time.time()

        reward = 0
        states = []

        for _ in range(self.action_repeat):
            put(vector_size, angle, speed, self.device_width, self.device_height)
            t = time.time()
            while True:
                frame  = self.asb.get_last_frame()
                if frame is not None:
                    break

            reward += self.get_reward(np.copy(frame))
            states.append(reward)
            done = self.is_done(np.copy(frame))

        next_state = []
        return next_state, reward, done, np.copy(frame)

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
            Currently working with fixed coords, not elegant though, futurely
            using object detection ...
        """
        restart_coords = self.device_ref_elements_data['try_again']['restart_btn_coords']
        self.episode_start_time = None
        tap(restart_coords[0], restart_coords[1])

if __name__ == "__main__":
    # Coords based on a 296x144 screen

    import random
    try_again_el_data = {
        'coords': [45, 60, 118, 180],
        'img_path': 'data/s8_cut_try_again.png',
        'restart_btn_coords': [640, 1110]
    }

    scores = {
        'coords_diamonds_gathered': [11, 27, 25, 35],
        'digits_mask_addr': '/home/neo/dev/balloma_rl_agent/misc/digits',
        'match_threshold': 10,
        'time_importance': 0.7,
        'diamonds_importance': 0.3,
        'episode_time_limit': 60,
        'diamonds_total': 7

    }

    env = Environment(device_ref_elements_data={'try_again': try_again_el_data,
                                                'scores': scores})

    # Actions generation
    exploration_mu = 0
    exploration_theta = 0.15
    exploration_sigma = 0.2
    action_size = 3
    action_low = np.array([1, 0, 1])
    action_high = np.array([10, 359, 2000])
    action_range = action_high - action_low

    #Start with random action
    action = np.array([np.random.uniform() for _ in action_low])
    noise = OUNoise(action.shape[0], exploration_mu,
                                 exploration_theta, exploration_sigma)

    for i in range(10):
        done = False
        j = 0
        while not done:
            j += 1
            ns = noise.sample()
            action = action + ns

            #print(action)
            #print(ns)
            v_size, angle, speed = np.array(transform_action(action, action_range, action_low),
                          dtype='uint8')
            ns, rw, done, frame = env.step(v_size, angle, speed)
            cv2.imshow('frame', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                #asb.stop()
                exit(0)
                break
            #cv2.imwrite(f'/tmp/frames/frame_dig_{j}.png', frame[11:27, 25:35])

        env.reset()

    cv2.destroyAllWindows()

