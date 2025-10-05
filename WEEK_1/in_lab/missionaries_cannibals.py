from collections import deque

def is_valid(state):
    missionaries, cannibals, boat = state
    if missionaries < 0 or cannibals < 0 or missionaries > 3 or cannibals > 3:
        return False
    if missionaries > 0 and missionaries < cannibals:
        return False
    if 3 - missionaries > 0 and 3 - missionaries < 3 - cannibals:
        return False
    return True

def get_successors(state):
    successors = []
    missionaries, cannibals, boat = state
    if boat == 1:
        moves = [(2, 0), (0, 2), (1, 1), (1, 0), (0, 1)]
        for move in moves:
            new_state = (missionaries - move[0], cannibals - move[1], 0)
            if is_valid(new_state):
                successors.append(new_state)
    else:
        moves = [(2, 0), (0, 2), (1, 1), (1, 0), (0, 1)]
        for move in moves:
            new_state = (missionaries + move[0], cannibals + move[1], 1)
            if is_valid(new_state):
                successors.append(new_state)
    return successors

def dfs(start_state, goal_state):
    queue = list([(start_state, [])])
    visited = set()
    while queue:
        (state, path) = queue.pop()
        if state in visited:
            continue
        visited.add(state)
        path = path + [state]
        if state == goal_state:
            return path, len(visited)
        for successor in get_successors(state):
            queue.append((successor, path))
    return None, len(visited)

def bfs(start_state, goal_state):
    queue = deque([(start_state, [])])
    visited = set()
    while queue:
        (state, path) = queue.popleft()
        if state in visited:
            continue
        visited.add(state)
        path = path + [state]
        if state == goal_state:
            return path, len(visited)
        for successor in get_successors(state):
            queue.append((successor, path))
    return None, len(visited)

start_state = (3, 3, 1)
goal_state = (0, 0, 0)

# DFS solution
solution_dfs, dfs_visited = dfs(start_state, goal_state)
print("\nDFS Solution:")
if solution_dfs:
    for step in solution_dfs:
        print(step)
else:
    print("No solution found.")
print(f"DFS visited nodes: {dfs_visited}")

# BFS solution
solution_bfs, bfs_visited = bfs(start_state, goal_state)
print("\nBFS Solution:")
if solution_bfs:
    for step in solution_bfs:
        print(step)
else:
    print("No solution found.")
print(f"BFS visited nodes: {bfs_visited}")

# Compare solutions based on visited nodes
print("\nComparison of DFS and BFS based on visited nodes:")
print(f"DFS visited nodes: {dfs_visited}")
print(f"BFS visited nodes: {bfs_visited}")
if dfs_visited < bfs_visited:
    print("DFS visited fewer nodes.")
elif bfs_visited < dfs_visited:
    print("BFS visited fewer nodes.")
else:
    print("Both DFS and BFS visited the same number of nodes.")