import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

MAX_BIKES = 20
MAX_MOVE = 5
GAMMA = 0.9
REWARD = 10
MOVE_COST = 2
PARKING_COST = 4
FREE_TRANSFER = 1
K = 12

def trunc(lam, K):
    p = [poisson.pmf(k, lam) for k in range(K)]
    t = 1 - sum(p)
    p.append(t)
    return np.array(p)

REQ = [3, 4]
RET = [3, 2]

p_r = [trunc(REQ[0], K), trunc(REQ[1], K)]
p_d = [trunc(RET[0], K), trunc(RET[1], K)]

states = [(i, j) for i in range(21) for j in range(21)]
idx = {s: i for i, s in enumerate(states)}
actions = list(range(-MAX_MOVE, MAX_MOVE + 1))

def trans(s, a):
    n1, n2 = states[s]
    if a > 0:
        free = min(FREE_TRANSFER, a)
        paid = a - free
    else:
        paid = abs(a)
    if a > n1 or -a > n2:
        return []
    n1a = n1 - a
    n2a = n2 + a
    mc = MOVE_COST * paid
    T = {}
    for r1 in range(K + 1):
        for r2 in range(K + 1):
            for d1 in range(K + 1):
                for d2 in range(K + 1):
                    p = p_r[0][r1] * p_r[1][r2] * p_d[0][d1] * p_d[1][d2]
                    if p < 1e-12:
                        continue
                    rent1 = min(n1a, r1)
                    rent2 = min(n2a, r2)
                    e1 = min(n1a - rent1 + d1, MAX_BIKES)
                    e2 = min(n2a - rent2 + d2, MAX_BIKES)
                    pen = 0
                    if e1 > 10:
                        pen += PARKING_COST
                    if e2 > 10:
                        pen += PARKING_COST
                    rwd = REWARD * (rent1 + rent2) - mc - pen
                    sp = idx[(e1, e2)]
                    if sp not in T:
                        T[sp] = [0.0, rwd]
                    T[sp][0] += p
    return [(sp, p, r) for sp, (p, r) in T.items()]

def policy_iteration():
    N = len(states)
    pi = np.zeros(N, dtype=int)
    V = np.zeros(N)
    t = 1e-3
    stable = False
    while not stable:
        while True:
            d = 0
            for s in range(N):
                a = actions[pi[s]]
                L = trans(s, a)
                if not L:
                    continue
                v = 0
                for sp, p, r in L:
                    v += p * (r + GAMMA * V[sp])
                d = max(d, abs(v - V[s]))
                V[s] = v
            if d < t:
                break
        stable = True
        for s in range(N):
            old = pi[s]
            best = old
            q_best = -1e18
            for ai, a in enumerate(actions):
                L = trans(s, a)
                if not L:
                    continue
                q = 0
                for sp, p, r in L:
                    q += p * (r + GAMMA * V[sp])
                if q > q_best:
                    q_best = q
                    best = ai
            pi[s] = best
            if best != old:
                stable = False
    return pi, V

pi3, V3 = policy_iteration()

policy_grid = np.zeros((21, 21))
value_grid = np.zeros((21, 21))

for (n1, n2), s in idx.items():
    policy_grid[n1, n2] = actions[pi3[s]]
    value_grid[n1, n2] = V3[s]

plt.figure(figsize=(7, 6))
plt.imshow(policy_grid, origin="lower")
plt.colorbar()
plt.xlabel("Location 2")
plt.ylabel("Location 1")
plt.title("Optimal Transfer Policy")
plt.tight_layout()
plt.savefig("gbike_policy_prob3.png", dpi=300)
plt.close()

plt.figure(figsize=(7, 6))
plt.imshow(value_grid, origin="lower")
plt.colorbar()
plt.xlabel("Location 2")
plt.ylabel("Location 1")
plt.title("Optimal Value Function")
plt.tight_layout()
plt.savefig("gbike_value_prob3.png", dpi=300)
plt.close()