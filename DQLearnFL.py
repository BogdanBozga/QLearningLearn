import numpy as np


class AgentDQFL:
    def __init__(self, lr, gamma, n_actions, n_states, eps_start, eps_end, eps_dec):
        self.lr = lr
        self.gamma = gamma
        self.n_actions = n_actions
        self.n_states = n_states
        self.eps_start = eps_start
        self.epsilon = self.eps_start
        self.eps_end = eps_end
        self.eps_dec = eps_dec

        self.Qa = {}
        self.Qb = {}
        self.init_Q()

    def init_Q(self):
        for state in range(self.n_states):
            for action in range(self.n_actions):
                self.Qa[(state, action)] = 0.0
                self.Qb[(state, action)] = 0.0
        print(self.Qa)

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.choice([i for i in range(self.n_actions)])
        else:
            action = 0
            value = 0
            for a in range(self.n_actions):
                if np.random.random() < 0.5:
                    value_l = self.Qa[(state, a)]
                else:
                    value_l = self.Qb[(state, a)]
                if value < value_l:
                    value = value_l
                    action = a
        return action

    def decrease_epsilon(self):
        self.epsilon = self.epsilon*self.eps_dec if self.epsilon > self.eps_end else self.epsilon

    def learn(self, state, action, reward, new_state):
        a_max = 0
        value = 0
        updateaA = True if np.random.random() < 0.5 else False
        for a in range(self.n_actions):
            if updateaA:
                value_l = self.Qa[(new_state, a)]
            else:
                value_l = self.Qb[(new_state, a)]
            if value < value_l:
                value = value_l
                a_max = a
        if updateaA:
            self.Qa[(state, action)] += self.lr*(
                        reward+self.gamma*self.Qa[new_state, a_max] - self.Qa[(state, action)])
        else:
            self.Qb[(state, action)] += self.lr * (
                        reward + self.gamma * self.Qb[new_state, a_max] - self.Qb[(state, action)])
        self.decrease_epsilon()
