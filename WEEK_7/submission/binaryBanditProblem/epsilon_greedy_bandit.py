import numpy as np
import matplotlib.pyplot as plt

class EpsilonGreedyBandit:
    def __init__(self, epsilon, n_actions):
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.q_values = np.zeros(n_actions) 
        self.action_counts = np.zeros(n_actions)

    def selectAction(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.q_values)

    def updateQValue(self, action, reward):
        self.action_counts[action] += 1
        self.q_values[action] += (reward - self.q_values[action]) / self.action_counts[action]

def simulateBanditForOne(bandit_func, epsilon, n_episodes):
    bandit = EpsilonGreedyBandit(epsilon, 1)  
    rewards = []

    for episode in range(n_episodes):
        action = bandit.selectAction() 
        reward = bandit_func()  
        bandit.updateQValue(action, reward)  
        rewards.append(reward)

    return rewards, bandit.q_values