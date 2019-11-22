import numpy as np
from control import put, get_timelapse_frame, on_game
import time


class Environment():
    """Environment, defines the goal and provides feedback to the agent."""
    def __init__(self):
        self.action_repeat = 3
        self.state_frame_width = 240
        self.state_frame_height = 240
        self.state_size = self.action_repeat * self.state_frame_width * self.state_frame_height

    def get_reward(self, state_frame):
        """
            Given a state frame extracts reward
            involved view components, converts it to
            float representation and computes reward.
        """
        # TODO:
        return np.random.random()

    def step(self, vector_size, angle, speed):
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
            put(vector_size, angle, speed)
            t = time.time()
            frame = get_timelapse_frame(t, self.state_frame_width,
                                        self.state_frame_height)
            reward += self.get_reward(frame)
            states.append(reward)
            done = not on_game(frame)
        next_state = np.concatenate(states)
        return next_state, reward, done

    def reset(self):
        """
            Put the environment back on game
        """
        # TODO:
        return True
