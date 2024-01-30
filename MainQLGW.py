from GridWorld import GridWorld
import numpy as np
import matplotlib.pyplot as plt
from QLearnFL import AgentQFL
from DQLearnFL import AgentDQFL
from NQLeanFL import AgentNQFL

def run_agent(agent, env, n_games,img_name):
    scores = []
    win_pcts = []
    nr_steps_l = []
    print('reset', env.reset())
    print('step', env.step(1))
    for i in range(n_games):
        done = False
        state = env.reset()
        score = 0
        nr_steps = 0
        while not done:
            action = agent.choose_action(state)
            new_state, reward, done, nr_steps = env.step(action)

            agent.learn(state, action, reward, new_state)
            score += reward
            state = new_state
        nr_steps_l.append(nr_steps)
        scores.append(score)
        if i % 100 == 0:
            win_pct = np.mean(scores[-100:])
            win_pcts.append(win_pct)
            if i % 1000 == 0:
                print('episode ', i, 'win pct %.2f' % win_pct, 'epsilon %.2f' % agent.epsilon)
    print(f"Average number of steps: {sum(nr_steps_l)/len(nr_steps_l)}")
    print(f"Average score: {sum(scores) / len(scores)}")
    plt.plot(nr_steps_l)
    plt.savefig(f"plots/{img_name}")
    plt.show()



if __name__ == "__main__":
    env = GridWorld()
    nr_qs = 10
    agentQ = AgentQFL(lr=0.1, gamma=g0.95, eps_start=1.0, eps_end=0.01, eps_dec=0.999999, n_actions=4, n_states=8*8)
    agentDQ = AgentDQFL(lr=0.1, gamma=0.95, eps_start=1.0, eps_end=0.01, eps_dec=0.999999, n_actions=4, n_states=8*8)
    agentNQ = AgentNQFL(lr=0.1, gamma=0.95, eps_start=1.0, eps_end=0.01, eps_dec=0.999999, n_actions=4, n_states=8*8,nq=nr_qs)

    # run_agent(agentQ, env, 100000,"QL-GridWorld.png")
    # run_agent(agentDQ, env, 100000,"DQL-GridWorld.png")
    run_agent(agentNQ, env, 100000*(nr_qs//2), f"{nr_qs}QL-GridWorld.png")