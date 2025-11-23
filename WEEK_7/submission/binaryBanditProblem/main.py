import numpy as np
import matplotlib.pyplot as plt
from epsilon_greedy_bandit import simulateBanditForOne
from binaryBanditA import binaryBanditA
from binaryBanditB import binaryBanditB

if __name__ == "__main__":
    epsilon = 0.1
    episodes = 1000
    rewards_A, q_values_A = simulateBanditForOne(binaryBanditA, epsilon, episodes)
    rewards_B, q_values_B = simulateBanditForOne(binaryBanditB, epsilon, episodes)

    print(f"Estimated q-values for Bandit A: {q_values_A}")
    print(f"Total rewards for Bandit A: {sum(rewards_A)}")

    print(f"Estimated q-values for Bandit B: {q_values_B}")
    print(f"Total rewards for Bandit B: {sum(rewards_B)}")

    cumulative_rewards_A = np.cumsum(rewards_A)
    expected_rewards_A = cumulative_rewards_A / np.arange(1, len(rewards_A) + 1)

    cumulative_rewards_B = np.cumsum(rewards_B)
    expected_rewards_B = cumulative_rewards_B / np.arange(1, len(rewards_B) + 1)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(expected_rewards_A, label='Expected Reward (Bandit A)', color='blue')
    plt.xlabel('Episodes')
    plt.ylabel('Expected Reward')
    plt.title('Expected Rewards for Bandit A')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(expected_rewards_B, label='Expected Reward (Bandit B)', color='green')
    plt.xlabel('Episodes')
    plt.ylabel('Expected Reward')
    plt.title('Expected Rewards for Bandit B')
    plt.grid(True)

    plt.tight_layout()
    plt.show()