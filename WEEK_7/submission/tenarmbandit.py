import numpy as np
import matplotlib.pyplot as plt

class NonStationaryBandit:
    def __init__(self, arms=10, epsilon=0.1, steps=1000):
        self.arms = arms
        self.epsilon = epsilon
        self.steps = steps
        self.q_values = np.zeros(arms)  
        self.action_counts = np.zeros(arms)  
        self.mean_rewards = np.zeros(arms)  
        self.total_rewards = []
        
    def randomWalk(self):
        self.mean_rewards += np.random.normal(0, 0.01, self.arms)

    def epsilon_greedy_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.arms)
        else:
            return np.argmax(self.q_values)

    def step(self):
        action = self.epsilon_greedy_action()
        reward = np.random.normal(self.mean_rewards[action], 1)  
        self.action_counts[action] += 1
        self.q_values[action] += (reward - self.q_values[action]) / self.action_counts[action]  
        
        self.randomWalk() 
        self.total_rewards.append(reward)
        return action, reward

    def run(self):
        for _ in range(self.steps):
            self.step()
        return self.q_values, self.total_rewards

bandit = NonStationaryBandit(arms = 10, epsilon = 0.1, steps = 10000)
final_q_values, rewards = bandit.run()
print("Q-values:", final_q_values)
print("Total reward collected:", sum(rewards))

cumulative_rewards = np.cumsum(rewards)
average_rewards = cumulative_rewards / np.arange(1, len(rewards) + 1)
plt.plot(average_rewards, color='red')
plt.xlabel('No. of steps')
plt.ylabel('Expected Reward')
plt.title('Expected Reward vs Steps')
plt.grid(True)
plt.show()