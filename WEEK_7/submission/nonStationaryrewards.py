import numpy as np
import matplotlib.pyplot as plt

class NonStationaryBandit:
    def __init__(self, arms=10, steps=10000):
        self.arms = arms
        self.steps = steps
        self.mean_rewards = np.zeros(arms)  
        self.total_rewards = []
        
    def random_walk(self):
        self.mean_rewards += np.random.normal(0, 0.01, self.arms)
    
    def get_reward(self, action):
        return np.random.normal(self.mean_rewards[action], 1)
    
    def step(self, action):
        reward = self.get_reward(action)
        self.random_walk() 
        self.total_rewards.append(reward)
        return reward

class ModifiedEpsilonGreedyAgent:
    def __init__(self, arms=10, epsilon=0.1, alpha=0.1):
        self.arms = arms
        self.epsilon = epsilon
        self.alpha = alpha  
        self.q_values = np.zeros(arms)  
        self.action_counts = np.zeros(arms) 
    
    def select_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.arms)  
        else:
            return np.argmax(self.q_values) 

    def update_estimates(self, action, reward):
        self.q_values[action] += self.alpha * (reward - self.q_values[action])
        self.action_counts[action] += 1

def run_experiment(bandit, agent, steps=10000):
    rewards = np.zeros(steps)
    actions = np.zeros(steps)
    
    for step in range(steps):
        action = agent.select_action()
        reward = bandit.step(action)
        agent.update_estimates(action, reward)
        
        rewards[step] = reward
        actions[step] = action
    
    return rewards, actions

bandit = NonStationaryBandit(arms=10, steps=10000)
agent = ModifiedEpsilonGreedyAgent(arms=10, epsilon=0.1, alpha=0.7)

rewards, actions = run_experiment(bandit, agent, 10000)

plt.figure(figsize=(10, 5))

cumulative_rewards = np.cumsum(rewards)
average_rewards = cumulative_rewards / (np.arange(1, 10000 + 1))
plt.plot(average_rewards, color='blue', label='Average Reward')
plt.xlabel('Steps')
plt.ylabel('Average Reward')
plt.title('Modified Epsilon-Greedy Agent in Non-Stationary Environment')
plt.legend()
plt.grid(True)
plt.show()