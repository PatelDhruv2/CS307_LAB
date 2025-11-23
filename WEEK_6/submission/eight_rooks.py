import numpy as np

class HopfieldNetwork:
    def __init__(self, n):
        self.n = n
        self.num_units = n * n  
        self.weights = np.zeros((self.num_units, self.num_units))
        self.thresholds = np.full(self.num_units, -1)

        self.initialize_weights()

    def initialize_weights(self):
        for i in range(self.n):
            for j in range(self.n):
                idx = i * self.n + j  

                for k in range(self.n):
                    if k != j:  
                        row_neighbor_idx = i * self.n + k
                        self.weights[idx][row_neighbor_idx] = -2

                for k in range(self.n):
                    if k != i:  
                        col_neighbor_idx = k * self.n + j
                        self.weights[idx][col_neighbor_idx] = -2

    # def energy(self, state):
    #     state = state.reshape(-1, 1)  
    #     energy = -0.5 * np.sum(self.weights * np.dot(state, state.T)) - np.dot(self.thresholds, state.flatten())
    #     return energy

    def update_state(self, state):
        for i in range(self.num_units):
            input_sum = np.dot(self.weights[i], state) - self.thresholds[i]
            state[i] = 1 if input_sum > 0 else 0  
        return state

    def solve(self, max_iterations=100):
        state = np.random.choice([0, 1], size=self.num_units)

        for _ in range(max_iterations):
            prev_state = state.copy()
            state = self.update_state(state)

            if np.array_equal(prev_state, state):
                break

        return state

n = 8  
hopfield = HopfieldNetwork(n)
solution = hopfield.solve()
solution_board = solution.reshape((n, n))  

print("Solution board:")
print(solution_board)