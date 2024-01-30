import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from QLearnFL import AgentQFL
from DQLearnFL import AgentDQFL
from NQLeanFL import AgentNQFL

def run_agent(env,n_games, Qagent, img_name):
    scores = []
    win_pcts = []
    win_nr = []
    print('reset', env.reset())
    print('step', env.step(1))
    for i in range(n_games):
        done = False
        state, info = env.reset()
        score = 0
        while not done:
            action = Qagent.choose_action(state)
            new_state, reward, done, truncated, info = env.step(action)

            Qagent.learn(state, action, reward, new_state)
            score += reward
            state = new_state
        scores.append(score)
        if i % 200 == 0:
            win_pct = np.mean(scores[-200:])
            win_nr.append(i)
            win_pcts.append(win_pct)
            if i % 200 == 0:
                print('episode ', i, 'win pct %.2f' % win_pct, 'epsilon %.2f' % Qagent.epsilon)

    plt.plot(win_nr, win_pcts)
    plt.savefig(f"plots/{img_name}")
    plt.show()

if __name__ == "__main__":
    env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=True)
    agentQ = AgentQFL(lr=0.1, gamma=0.95, eps_start=1.0, eps_end=0.01, eps_dec=0.999999, n_actions=4, n_states=8*8)
    agentDQ = AgentDQFL(lr=0.1, gamma=0.95, eps_start=1.0, eps_end=0.01, eps_dec=0.999999, n_actions=4, n_states=8*8)

    nr_qs = 10
    agentNQ = AgentNQFL(lr=0.1, gamma=0.95, eps_start=1.0, eps_end=0.01, eps_dec=0.999999, n_actions=4, n_states=8*8,nq=nr_qs)

    img_name = "QL-FrozenLake.png"
    run_agent(env, 10000*(nr_qs//2), agentQ, img_name)