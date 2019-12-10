import time

def train(agent, env, num_episodes=1000, episode_seconds_constrain=None):
    for i_episode in range(1, num_episodes+1):
        episode_start_time = time.time()
        state = agent.reset_episode() # start a new episode
        while True:
            action = agent.act(state)
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
                state = agent.reset_episode()
                break

if __name__ == "__main__":
    from ddpg import DDPG
    from task import Environment
    env = Environment()
    #agent = DDPG()
