import time
import numpy as np

def train(agent, env, num_episodes=1000, episode_seconds_constrain=None):
    for i_episode in range(1, num_episodes+1):
        state = agent.reset_episode() # start a new episode
        episode_start_time = time.time()
        while True:
            action = agent.act(state)
            print("Action is:")
            print(np.shape(action))

            next_state, reward, done = env.step(action)
            t = time.time() - episode_start_time
            if episode_seconds_constrain is not None and t > episode_seconds_constrain:
                done = True
            agent.step(action, reward, next_state, done)
            state = next_state
            if done:
                print("\rEpisode = {:4d}, score = {:7.3f} (best = {:7.3f})".format(i_episode, agent.score,
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

