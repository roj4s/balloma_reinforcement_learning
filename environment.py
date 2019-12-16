import numpy as np
from control import put, match_imgs, tap, transform_action
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
        self.action_low = np.array([1, 0, 1])
        self.action_high = np.array([10, 359, 2000])
        self.action_size = self.action_low.shape[0]
        self.action_range = self.action_high - self.action_low
        self.device_ref_elements_data = device_ref_elements_data
        self.visible_state_area_data = self.device_ref_elements_data['scores']['state_area']
        self.state_frame_width = self.visible_state_area_data[3] - self.visible_state_area_data[2]
        self.state_frame_height = self.visible_state_area_data[1] - self.visible_state_area_data[0]
        self.device_evt = Event()
        self.state_size = (self.state_frame_height,
                           self.state_frame_width, 3 * self.action_repeat)
        self.digits_matcher = DigitsMatcher(self.device_ref_elements_data['scores']['digits_mask_addr'])
        self.asb.run()
        self.episode_start_time = None

    def get_reward(self, frame):
        """
            Given a state frame extracts reward
            involved view components, converts it to
            float representation and computes reward.
        """
        t = time.time() - self.episode_start_time
        done_comparison_data = self.device_ref_elements_data['done_comparison_data']
        coords_done_success = done_comparison_data['coords_done_success']
        img_done_success = done_comparison_data['img_done_success']
        done_succ = match_imgs(frame, coords_done_success, img_done_success)
        if done_succ:
            return 2

        img_done_fail = done_comparison_data['img_done_fail']
        coords_done_fail = done_comparison_data['coords_done_fail']
        done_fail = match_imgs(frame, coords_done_fail, img_done_fail)
        if done_fail:
            return -2


        match_threshold = self.device_ref_elements_data['scores']['match_threshold']
        episode_time_limit = self.device_ref_elements_data['scores']['episode_time_limit']
        diamonds_total = self.device_ref_elements_data['scores']['diamonds_total']
        diamonds_importance = self.device_ref_elements_data['scores']['diamonds_importance']
        time_importance = self.device_ref_elements_data['scores']['time_importance']
        y1, y2, x1, x2 = self.device_ref_elements_data['scores']['coords_diamonds_gathered']
        cropped_dig = frame[y1:y2, x1:x2]
        diamonds_gathered = self.digits_matcher.match(cropped_dig,
                                                      threshold=match_threshold)

        if diamonds_gathered is None:
            return 0

        return diamonds_importance * (diamonds_gathered/diamonds_total) - time_importance * t/episode_time_limit


    def step(self, action):
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
        next_state = np.zeros(self.state_size)
        '''
        vector_size, angle, speed = np.array(transform_action(action,
                                                              self.action_range,
                                                              self.action_low),
                          dtype='uint8')

        '''
        vector_size, angle, speed = action[0], action[1], action[2]

        for i in range(0, self.action_repeat * 3, 3):
            put(vector_size, angle, speed, self.device_width, self.device_height)

            time.sleep(0.5)
            while True:
                frame  = self.asb.get_last_frame()
                if frame is not None:
                    break

            #cv2.imwrite(f'/tmp/frames/frame_{j}.png', np.copy(frame))

            reward += self.get_reward(np.copy(frame))
            next_state[:, :, i:i+3] = self.get_state_from_frame(np.copy(frame))

        done = self.is_done(np.copy(frame))

        return next_state, reward, done

    def get_state_from_frame(self, frame):
        y1, y2, x1, x2 = self.visible_state_area_data
        return frame[y1: y2, x1: x2]


    def is_done(self, frame):
        '''
            Currently checking by comparing images.
            It will be replaced by a less device-dependent approach.
        '''
        done_comparison_data = self.device_ref_elements_data['done_comparison_data']
        coords_done_fail = done_comparison_data['coords_done_fail']
        coords_done_success = done_comparison_data['coords_done_success']
        img_done_fail = done_comparison_data['img_done_fail']
        img_done_success = done_comparison_data['img_done_success']
        return match_imgs(frame, coords_done_fail, img_done_fail) or match_imgs(frame, coords_done_success, img_done_success)

    def reset(self):
        """
            Put the environment back on game
            Currently working with fixed coords, not elegant though, futurely
            using object detection ...
        """
        data = self.device_ref_elements_data['done_comparison_data']
        restart_coords = data['restart_btn_coords']
        self.episode_start_time = None
        tap(restart_coords[0], restart_coords[1])

        coords_restart_ongame = data['restart_ongame']
        for x, y in coords_restart_ongame:
            tap(x, y)
            time.sleep(1)

        state = np.zeros(self.state_size)
        while True:
            frame = self.asb.get_last_frame()
            if frame is not None:
                break


        time.sleep(1)

        for i in range(0, self.action_repeat * 3, 3):
            state[:, :, i:i+3] = self.get_state_from_frame(np.copy(frame))

        return state


if __name__ == "__main__":
    # Coords based on a 296x144 frames

    import random
    done_comparison_data = {
        'coords_done_fail': [45, 60, 118, 180],
        'coords_done_success': [5, 16, 122, 174],
        'img_done_fail': 'data/s8_cut_try_again.png',
        'img_done_success': '/home/neo/dev/balloma_rl_agent/data/game_score_s8.png',
        'restart_btn_coords': [640, 1110],
        'restart_ongame': [(2764, 93), (2624, 552)],
    }

    scores = {
        'coords_diamonds_gathered': [11, 27, 25, 35],
        'digits_mask_addr': '/home/neo/dev/balloma_rl_agent/misc/digits',
        'match_threshold': 10,
        'state_area': [28, 112, 0, 296],
        'time_importance': 0.7,
        'diamonds_importance': 0.3,
        'episode_time_limit': 60,
        'diamonds_total': 7

    }

    env = Environment(device_ref_elements_data={'done_comparison_data':
                                                done_comparison_data,
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



    time_limit = 10
    for i in range(10):
        start_time = time.time()
        env.reset()
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
            ns, rw, done = env.step((v_size, angle, speed))
            if time.time() - start_time > time_limit:
                print("Time limit, will reset")
                done = True
                #env.reset()

            #cv2.imshow('frame', frame[28: 112, :])
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                #asb.stop()
                exit(0)
                break


    cv2.destroyAllWindows()

