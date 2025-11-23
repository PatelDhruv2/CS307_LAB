import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

MAX_BIKES = 20
MAX_MOVE = 5
GAMMA = 0.9
REWARD = 10
MOVE_COST = 2
K = 11

def poisson_probs(lam, K):
    p = [poisson.pmf(k, lam) for k in range(K)]
    t = 1 - sum(p)
    p.append(t)
    return np.array(p)

REQ = [3, 4]
RET = [3, 2]

p_r1 = poisson_probs(REQ[0], K)
p_r2 = poisson_probs(REQ[1], K)
p_d1 = poisson_probs(RET[0], K)
p_d2 = poisson_probs(RET[1], K)

states = [(i, j) for i in range(MAX_BIKES + 1) for j in range(MAX_BIKES + 1)]
state_to_idx = {s: i for i, s in enumerate(states)}
N = len(states)
actions = list(range(-MAX_MOVE, MAX_MOVE + 1))

def compute_transitions():
    TR = [[[] for _ in actions] for _ in range(N)]
    for si, (n1, n2) in enumerate(states):
        for ai, a in enumerate(actions):
            if a > n1 or -a > n2:
                continue
            n1_after = n1 - a
            n2_after = n2 + a
            mc = MOVE_COST * abs(a)
            trans = {}
            for r1 in range(K + 1):
                for r2 in range(K + 1):
                    for d1 in range(K + 1):
                        for d2 in range(K + 1):
                            p = p_r1[r1] * p_r2[r2] * p_d1[d1] * p_d2[d2]
                            if p < 1e-8:
                                continue
                            rent1 = min(n1_after, r1)
                            rent2 = min(n2_after, r2)
                            rwd = REWARD * (rent1 + rent2) - mc
                            e1 = min(n1_after - rent1 + d1, MAX_BIKES)
                            e2 = min(n2_after - rent2 + d2, MAX_BIKES)
                            j = state_to_idx[(e1, e2)]
                            if j not in trans:
                                trans[j] = [0.0, rwd]
                            trans[j][0] += p
            for sp, (p, r) in trans.items():
                TR[si][ai].append((sp, p, r))
    return TR

TR = compute_transitions()

def policy_iteration():
    pi = np.zeros(N, dtype=int)
    V = np.zeros(N)
    theta = 1e-4
    stable = False
    while not stable:
        while True:
            d = 0
            for s in range(N):
                a = pi[s]
                v = 0
                for sp, p, r in TR[s][a]:
                    v += p * (r + GAMMA * V[sp])
                d = max(d, abs(v - V[s]))
                V[s] = v
            if d < theta:
                break
        stable = True
        for s in range(N):
            old = pi[s]
            best = old
            q_best = -1e12
            for ai, a in enumerate(actions):
                q = 0
                for sp, p, r in TR[s][ai]:
                    q += p * (r + GAMMA * V[sp])
                if q > q_best:
                    q_best = q
                    best = ai
            pi[s] = best
            if best != old:
                stable = False
    return pi, V

pi, V = policy_iteration()

policy_grid = np.zeros((MAX_BIKES + 1, MAX_BIKES + 1))
value_grid = np.zeros((MAX_BIKES + 1, MAX_BIKES + 1))

for (n1, n2), s in state_to_idx.items():
    policy_grid[n1, n2] = actions[pi[s]]
    value_grid[n1, n2] = V[s]

plt.figure(figsize=(7, 6))
plt.imshow(policy_grid, origin="lower")
plt.colorbar()
plt.xlabel("Location 2")
plt.ylabel("Location 1")
plt.title("Optimal Transfer Policy")
plt.tight_layout()
plt.savefig("gbike_policy_prob2.png", dpi=300)
plt.close()

plt.figure(figsize=(7, 6))
plt.imshow(value_grid, origin="lower")
plt.colorbar()
plt.xlabel("Location 2")
plt.ylabel("Location 1")
plt.title("Optimal Value Function")
plt.tight_layout()
plt.savefig("gbike_value_prob2.png", dpi=300)
plt.close()