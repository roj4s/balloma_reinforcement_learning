import time
import numpy as np
import os

def train(agent, env, num_episodes=10000000000000000000,
          episode_seconds_constrain=None, output_path=None):

    timestmp = time.time()

    if output_path is None:
        output_path = '/home/neo/dev/balloma_rl_agent/outputs'

    log_output = os.path.join(output_path, f"log_{timestmp}")

    columns = ('episode', 'step', 'reward', 'loss', 'done', 'timestamp',
               'vector_size', 'angle', 'speed')
    col_frm = ",".join("{}" for _ in columns)
    col_frm += '\n'

    with open(log_output, 'wt') as f:
        f.write(col_frm.format(*columns))

    for i_episode in range(1, num_episodes+1):
        state = agent.reset_episode() # start a new episode
        episode_start_time = time.time()
        step_i = 0
        run = True
        while run:
            step_timestamp = time.time()
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            print(f"Episode: {i_episode}, Step: {step_i}, Reward: {reward}, Done: {done}")
            t = step_timestamp - episode_start_time

            if episode_seconds_constrain is not None and t > episode_seconds_constrain:
                run = False

            agent.step(action, reward, next_state, done)
            state = next_state

            with open(log_output, 'at') as f:
                f.write(col_frm.format(i_episode, step_i, reward,
                                       agent.last_loss, done, step_timestamp,
                                       *action))

            step_i += 1
            #agent.save_data(output_path, _id=timestmp)

            if done:
                agent_memory_len = len(agent.memory)
                print("\rEpisode = {:4d}, Experiences: {},  score = {:7.3f}|"\
                      "(best = {:7.3f})".format(i_episode, agent_memory_len, agent.score,
                                               agent.best_score,
                                               ), end="")

                break

        if len(agent.memory) > agent.batch_size:
                    experiences = agent.memory.sample()
                    agent.learn(experiences)

if __name__ == "__main__":
    from agent import DDPG, DeepQAgent
    from environment import Environment

    done_comparison_data = {
        'coords_done_fail': [45, 60, 118, 180],
        'coords_done_success': [5, 16, 122, 174],
        'img_done_fail': 'data/s8_cut_try_again.png',
        'img_done_success': 'data/game_score_s8.png',
        'restart_btn_coords': [640, 1110],
        'restart_ongame': [(2764, 93), (2624, 552)],
    }

    scores = {
        'coords_diamonds_gathered': [11, 27, 25, 35],
        'digits_mask_addr': 'data/digits',
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
    #agent = DDPG(env)
    agent = DeepQAgent(env)
    train(agent, env, episode_seconds_constrain=45)

