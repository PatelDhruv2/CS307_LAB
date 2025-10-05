import numpy as np
import matplotlib.pyplot as plt
import random
import math

# Simulated Annealing applied to Jigsaw Puzzle Reconstruction
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
    plt.title("Reconstructed Image (Simulated Annealing)")
    plt.show()

def calc(state, patches):
    total = 0
    for i in range(4):
        for j in range(4):
            if j + 1 < 4:
                for k in range(len(patches[state[i][j]])):
                    total += abs(
                        patches[state[i][j]][k][127] -
                        patches[state[i][j+1]][k][0]
                    )
            if i + 1 < 4:
                for k in range(len(patches[state[i][j]])):
                    total += abs(
                        patches[state[i][j]][127][k] -
                        patches[state[i+1][j]][0][k]
                    )
    return total

def neighbor(state):
    i1, j1 = random.randint(0, 3), random.randint(0, 3)
    i2, j2 = random.randint(0, 3), random.randint(0, 3)
    new_state = [row[:] for row in state]
    new_state[i1][j1], new_state[i2][j2] = new_state[i2][j2], new_state[i1][j1]
    return new_state

patches, state_mat = create_patch(load_matrix("scrambled_lena.mat").T)
state = state_mat
val = calc(state, patches)

T0 = 1000
for step in range(1000):
    T = T0 / (step + 1)
    new_state = neighbor(state)
    new_val = calc(new_state, patches)
    if new_val < val:
        state, val = new_state, new_val
    else:
        prob = math.exp((val - new_val) / T)
        if random.random() < prob:
            state, val = new_state, new_val

final_img = reconstruct_image(patches, state)
print_image(final_img)