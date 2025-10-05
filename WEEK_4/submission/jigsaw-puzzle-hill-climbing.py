import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque

# Load scrambled Lena image
def load_matrix(file_path):
    matrix = []
    with open(file_path, "r") as f:
        lines = f.readlines()[1:]
    for line in lines:
        line = line.strip()
        if line:
            matrix.append(int(line))
    matrix = np.array(matrix)
    return matrix.reshape((512, 512))

def create_patch(image):
    patches, state_mat = [], []
    z = 0
    for i in range(4):
        temp = []
        for j in range(4):
            temp.append(z)
            patch = image[i*128:(i+1)*128, j*128:(j+1)*128]
            patches.append(patch)
            z += 1
        state_mat.append(temp)
    return patches, state_mat

def reconstruct_image(patches, grid):
    ph, pw = patches[0].shape[:2]
    img = np.zeros((4*ph, 4*pw), dtype=patches[0].dtype)
    for i in range(4):
        for j in range(4):
            img[i*ph:(i+1)*ph, j*pw:(j+1)*pw] = patches[grid[i][j]]
    return img

def print_image(image):
    plt.imshow(image, cmap="gray")
    plt.title("Reconstructed Image (Hill Climbing)")
    plt.show()

def get_successors(state, patches, visited, vertical):
    succ = []
    for i, mat in enumerate(patches):
        if i in visited:
            continue
        diff = 0
        if vertical:
            for j in range(len(mat)):
                diff += abs(patches[state][-1][j] - mat[0][j])
        else:
            for j in range(len(mat)):
                diff += abs(patches[state][j][-1] - mat[j][0])
        succ.append((diff, i))
    succ.sort()
    best_diff = succ[0][0]
    ties = [x for x in succ if x[0] == best_diff]
    return random.choice(ties)

def traversal(start, patches):
    visited, queue = [], deque([(start, 0)])
    visited.append(start)
    temp = [0] * 16
    total = 0
    while queue:
        node, pos = queue.popleft()
        temp[pos] = node
        if pos + 4 < 16:
            nxt = get_successors(node, patches, visited, True)
            queue.append((nxt[1], pos + 4))
            visited.append(nxt[1])
            total += nxt[0]
        if pos % 4 != 3:
            nxt = get_successors(node, patches, visited, False)
            queue.append((nxt[1], pos + 1))
            visited.append(nxt[1])
            total += nxt[0]
    new_state = [temp[i:i + 4] for i in range(0, 16, 4)]
    return new_state, total

def rearranged(patches):
    scores = []
    configs = []
    for i in range(16):
        state, val = traversal(i, patches)
        scores.append((abs(val), i))
        configs.append((state, i))
    scores.sort()
    for conf in configs:
        if conf[1] == scores[0][1]:
            return conf[0]

patches, state_mat = create_patch(load_matrix("scrambled_lena.mat").T)
new_state = rearranged(patches)
new_image = reconstruct_image(patches, new_state)
print_image(new_image)