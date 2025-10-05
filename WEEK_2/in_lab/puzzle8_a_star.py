import heapq
import random

class Node:
    def __init__(self, state, parent=None, g=0, h=0):
        self.state = state
        self.parent = parent
        self.g = g
        self.h = h
        self.f = g + h

    def __lt__(self, other):
        return self.f < other.f

def heuristic(state, goal_state):
    return sum(1 for i, tile in enumerate(state) if tile != goal_state[i] and tile != 0)

def get_successors(node):
    successors = []
    index = node.state.index(0)
    row, col = divmod(index, 3)
    
    moves = []
    if row > 0: moves.append(-3)
    if row < 2: moves.append(3)
    if col > 0: moves.append(-1)
    if col < 2: moves.append(1)

    for move in moves:
        new_index = index + move
        new_state = node.state[:]
        new_state[index], new_state[new_index] = new_state[new_index], new_state[index]
        successors.append(Node(new_state, node, node.g + 1))
    
    return successors

def a_star_solver(start_state, goal_state):
    start_node = Node(start_state, g=0, h=heuristic(start_state, goal_state))
    frontier = []
    heapq.heappush(frontier, start_node)
    visited = set()

    while frontier:
        node = heapq.heappop(frontier)

        if node.state == goal_state:
            path = []
            while node:
                path.append(node.state)
                node = node.parent
            return path[::-1]

        visited.add(tuple(node.state))

        for succ in get_successors(node):
            succ.h = heuristic(succ.state, goal_state)
            succ.f = succ.g + succ.h
            if tuple(succ.state) not in visited:
                heapq.heappush(frontier, succ)
    
    return None

def random_goal(start_state, steps=20):
    node = Node(start_state)
    for _ in range(steps):
        node = random.choice(get_successors(node))
    return node.state

start_state = [1, 2, 3, 4, 5, 6, 7, 8, 0]
goal_state = random_goal(start_state, 20)

solution = a_star_solver(start_state, goal_state)

if solution:
    print("Solution found:")
    for step in solution:
        print(step)
else:
    print("No solution found.")
