import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from QLearnFL import AgentQFL
from DQLearnFL import AgentDQFL

if __name__ == "__main__":
    env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=True)
    agentQ = AgentQFL(lr=0.1, gamma=0.95, eps_start=1.0, eps_end=0.01, eps_dec=0.999999, n_actions=4, n_states=8*8)
    agentDQ = AgentDQFL(lr=0.1, gamma=0.95, eps_start=1.0, eps_end=0.01, eps_dec=0.999999, n_actions=4, n_states=8*8)

    scores = []
    win_pcts = []
    n_games = 100000
    print('reset', env.reset())
    print('step', env.step(1))
    for i in range(n_games):
        done = False
        state, info = env.reset()
        score = 0
        while not done:
            action = agentQ.choose_action(state)
            new_state, reward, done, truncated, info = env.step(action)

            agentQ.learn(state, action, reward, new_state)
            score += reward
            state = new_state
        scores.append(score)
        if i % 100 == 0:
            win_pct = np.mean(scores[-100:])
            win_pcts.append(win_pct)
            if i % 1000 == 0:
                print('episode ', i, 'win pct %.2f' % win_pct, 'epsilon %.2f' % agentQ.epsilon)
    plt.plot(win_pcts)
    plt.show()


    for i in range(n_games):
        done = False
        state, info = env.reset()
        score = 0
        while not done:
            action = agentDQ.choose_action(state)
            new_state, reward, done, truncated, info = env.step(action)

            agentDQ.learn(state, action, reward, new_state)
            score += reward
            state = new_state
        scores.append(score)
        if i % 100 == 0:
            win_pct = np.mean(scores[-100:])
            win_pcts.append(win_pct)
            if i % 1000 == 0:
                print('episode ', i, 'win pct %.2f' % win_pct, 'epsilon %.2f' % agentDQ.epsilon)
    plt.plot(win_pcts)
    plt.show()