import numpy as np

class HopfieldTSP:
    def __init__(self, num_cities, distances, gamma=500):
        self.num_cities = num_cities
        self.distances = distances  
        self.gamma = gamma
        self.num_units = num_cities * num_cities
        self.weights = np.zeros((self.num_units, self.num_units))
        self.thresholds = np.zeros(self.num_units)
        self.state = np.random.randint(0, 2, self.num_units)
        self.initialize_weights()

    def index(self, city, step):
        return city * self.num_cities + step

    def initialize_weights(self):
        for i in range(self.num_cities):
            for k in range(self.num_cities):
                idx = self.index(i, k)

                for j in range(self.num_cities):
                    if j != i:  
                        self.weights[idx, self.index(j, k)] -= self.gamma
                    if j != k:  
                        self.weights[idx, self.index(i, j)] -= self.gamma

                for j in range(self.num_cities):
                    if j != i:
                        next_step = (k + 1) % self.num_cities
                        self.weights[idx, self.index(j, next_step)] -= self.distances[i][j]

        self.thresholds.fill(-self.gamma / 2)

    def update_state(self):
        for i in range(self.num_units):
            input_sum = np.dot(self.weights[i], self.state) - self.thresholds[i]
            self.state[i] = 1 if input_sum > 0 else 0

    def get_path(self):
        path = []
        for k in range(self.num_cities):
            for i in range(self.num_cities):
                if self.state[self.index(i, k)] == 1:
                    path.append(i)
                    break
        return path

    def solve(self, max_iterations=1000):
        for _ in range(max_iterations):
            prev_state = self.state.copy()
            self.update_state()
            if np.array_equal(prev_state, self.state):  
                break
        return self.get_path()

if __name__ == "__main__":
    num_cities = 10
    np.random.seed(42)  
    distances = np.random.randint(1, 100, (num_cities, num_cities))
    np.fill_diagonal(distances, 0)  
    distances = (distances + distances.T) // 2 

    tsp_solver = HopfieldTSP(num_cities, distances)
    solution = tsp_solver.solve()
    print("TSP Solution (city visit order):", solution)