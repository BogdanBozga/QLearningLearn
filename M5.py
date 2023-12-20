import gym
import numpy as np
import matplotlib.pyplot as plt

#  LEFT = 0, down = 1 right=2 up 3
# AFFF
# FHFH
# FFFH
# HFFG

policy = {0:2, 1:2, 2:0, 3:0, 4:0, 5:0, 6:2, 7:2, 8:0, 9:0, 10:0, 11:0, 12:2, 13:2, 14:0, 15:0, 16:0}
env = gym.make('FrozenLake-v1')
n_games = 1000
win = []
scores = []

for i in range(n_games):
    done = False
    obs, _ = env.reset()
    score = 0
    while not done:
        action = policy[obs]
        obs, reward, done, info, _ = env.step(action)
        score += reward
    scores.append(score)
    if i % 10 == 0:
        average = np.mean(scores[-10:])
        win.append(average)
plt.plot(win)
plt.show()