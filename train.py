import time
import numpy as np

def train(agent, env, num_episodes=1000, episode_seconds_constrain=None):
    for i_episode in range(1, num_episodes+1):
        state = agent.reset_episode() # start a new episode
        episode_start_time = time.time()
        step_i = 0
        while True:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            print(f"Episode: {i_episode}, Step: {step_i}, Reward: {reward}, Done: {done}")
            t = time.time() - episode_start_time
            if episode_seconds_constrain is not None and t > episode_seconds_constrain:
                done = True
            agent.step(action, reward, next_state, done)
            state = next_state
            step_i += 1
            if done:
                agent_memory_len = len(agent.memory)
                print("\rEpisode = {:4d}, Experiences: {},  score = {:7.3f}|"\
                      "(best = {:7.3f})".format(i_episode, agent_memory_len, agent.score,
                                               agent.best_score,
                                               ), end="")
                break

if __name__ == "__main__":
    from ddpg import DDPG
    from task import Environment

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
    agent = DDPG(env)
    train(agent, env, episode_seconds_constrain=30)

