import numpy as np
from random import randint

class AgentNQFL:
    def __init__(self, lr, gamma, n_actions, n_states, eps_start, eps_end, eps_dec, nq=2):
        self.lr = lr
        self.gamma = gamma
        self.n_actions = n_actions
        self.n_states = n_states
        self.eps_start = eps_start
        self.epsilon = self.eps_start
        self.eps_end = eps_end
        self.eps_dec = eps_dec
        self.Qs = []
        self.nq = nq
        for i in range(nq):
            self.Qs.append({})
        self.init_Q()

    def init_Q(self):
        for state in range(self.n_states):
            for action in range(self.n_actions):
                for i in range(self.nq):
                    self.Qs[i][(state, action)] = 0.0
        # print(self.Qa)
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.choice([i for i in range(self.n_actions)])
        else:
            action = 0
            value = 0
            for a in range(self.n_actions):
                i = randint(0, self.nq-1)
                value_l = self.Qs[i][(state, a)]
                if value < value_l:
                    value = value_l
                    action = a
        return action

    def decrease_epsilon(self):
        self.epsilon = self.epsilon*self.eps_dec if self.epsilon > self.eps_end else self.epsilon

    def learn(self, state, action, reward, new_state):
        a_max = 0
        value = 0
        for a in range(self.n_actions):
            i = randint(0, self.nq - 1)
            value_l = self.Qs[i][(new_state, a)]
            if value < value_l:
                value = value_l
                a_max = a

        i = randint(0, self.nq - 1)
        self.Qs[i][(state, action)] += self.lr*(
                        reward+self.gamma*self.Qs[i][new_state, a_max] - self.Qs[i][(state, action)])

        self.decrease_epsilon()
