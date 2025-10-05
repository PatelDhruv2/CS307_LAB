from collections import deque

def get_successors(state):
    successors = []
    for i in range(7):
        if state[i] == 1:
            if(i+1<7):
                if(state[i+1] == 0):
                    new_state = list(state)
                    new_state[i] = 0
                    new_state[i+1] = 1
                    successors.append(tuple(new_state))
            if(i+2<7):
                if(state[i+2] == 0):
                    new_state = list(state)
                    new_state[i] = 0
                    new_state[i+2] = 1
                    successors.append(tuple(new_state))
        if state[i] == -1:
            if(i-1>=0):
                if(state[i-1] == 0):
                    new_state = list(state)
                    new_state[i] = 0
                    new_state[i-1] = -1
                    successors.append(tuple(new_state))
            if(i-2>=0):
                if(state[i-2] == 0):
                    new_state = list(state)
                    new_state[i] = 0
                    new_state[i-2] = -1
                    successors.append(tuple(new_state))
    return successors

def dfs(start_state, goal_state):
    stack = list([(start_state, [])])
    visited = set()
    while stack:
        (state, path) = stack.pop()
        if state in visited:
            continue
        visited.add(state)
        path = path + [state]
        if state == goal_state:
            print("Total nodes explored", len(visited))
            return path
        for successor in get_successors(state):
            stack.append((successor, path))
    return None

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
            print("Total nodes explored", len(visited))
            return path
        for successor in get_successors(state):
            queue.append((successor, path))
    return None

start_state = (1,1,1,0,-1,-1,-1)
goal_state = (-1,-1,-1,0,1,1,1)
print("DFS Solution")
solution = dfs(start_state, goal_state)
if solution:
    print("Solution found:")
    print("Number of steps:", len(solution)-1)
    for step in solution:
        print(step)
else:
    print("No solution found.")
dfs_steps=len(solution)
print("BFS Solution")
solution = bfs(start_state, goal_state)
bfs_steps = 0
if solution:
    print("Solution found:")
    print("Number of steps:", len(solution)-1)
    for step in solution:
        print(step)
else:
    print("No solution found.")
bfs_steps=len(solution)
if dfs_steps < bfs_steps:
    print("DFS is better")
elif bfs_steps < dfs_steps:
    print("BFS is better")
else:
    print("Both are equal")