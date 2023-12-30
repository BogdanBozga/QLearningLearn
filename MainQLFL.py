import gym
import numpy as np
import matplotlib.pyplot as plt
from QLearnFL import Agent

if __name__ == "__main__":
    env = gym.make('FrozenLake-v1')
    agent = Agent(lr=0.001, gamma=0.9, eps_start=1.0, eps_end=0.01, eps_dec=0.999995, n_actions=4, n_states=16)

    scores = []
    win_pcts = []
    n_games = 500000
    for i in range(n_games):
        done = False
        state, info = env.reset()
        score = 0
        while not done:
            action = agent.choose_action(state)
            # print('action', action)
            state_, reward, done, truncated, info = env.step(action)
            # print('state', state)
            # print('state_', state_)

            agent.learn(state, action, reward, state_)
            score += reward
            state = state_
        scores.append(score)
        if i % 100 == 0:
            win_pct = np.mean(scores[-100:])
            win_pcts.append(win_pct)
            if i % 1000 == 0:
                print('episode ', i, 'win pct %.2f' % win_pct, 'epsilon %.2f' % agent.epsilon)
    plt.plot(win_pcts)
    plt.show()
