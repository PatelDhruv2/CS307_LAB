import random, math
import numpy as np
import matplotlib.pyplot as plt

NOTE_NAMES = {60: 'S', 61: 'r', 64: 'G', 65: 'M', 67: 'P', 68: 'd', 71: 'N', 72: "S'"}
SCALE = sorted(NOTE_NAMES.keys())

PHRASES = [
    [61, 64],        # r G
    [64, 65, 67],    # G M P
    [68, 71, 72],    # d N S'
    [61, 60],        # r S
    [64, 61, 60]     # G r S
]

def seq_to_names(seq):
    return [NOTE_NAMES.get(n, str(n)) for n in seq]

def contains_phrase(seq, phrase):
    L = len(phrase)
    for i in range(len(seq)-L+1):
        if seq[i:i+L] == phrase:
            return True
    return False

def fitness(seq):
    score = 0.1 * len(seq)
    for i in range(len(seq)-1):
        interval = abs(seq[i+1]-seq[i])
        if interval <= 2:
            score += 0.5
        elif interval <= 5:
            score += 0.1
        else:
            score -= 0.5
    for p in PHRASES:
        if contains_phrase(seq, p):
            score += 3.0
    if seq[-1] in (60,72):
        score += 2.0
    if seq[-1] == 72 and seq[-2] == 71:
        score += 1.0
    repeats = sum(1 for i in range(len(seq)-1) if seq[i]==seq[i+1])
    score -= 0.2*repeats
    return score

def random_seq(length=16):
    return [random.choice(SCALE) for _ in range(length)]

def neighbor(seq):
    s = seq.copy()
    if random.random() < 0.6:
        idx = random.randrange(len(s))
        s[idx] = random.choice(SCALE)
    else:
        i,j = random.sample(range(len(s)),2)
        s[i],s[j] = s[j],s[i]
    return s

# Simulated annealing
L = 16
T0 = 5.0
alpha = 0.995
steps = 25000

current = random_seq(L)
best = current.copy()
best_score = fitness(best)
current_score = best_score
T = T0

for step in range(steps):
    cand = neighbor(current)
    cand_score = fitness(cand)
    d = cand_score - current_score
    if d > 0 or math.exp(d / T) > random.random():
        current = cand
        current_score = cand_score
        if cand_score > best_score:
            best = cand.copy()
            best_score = cand_score
    T *= alpha
    if T < 1e-4:
        T = T0

print("Best fitness:", best_score)
print("Melody (MIDI):", best)
print("Melody (Bhairav names):", seq_to_names(best))

phrase_hits = {tuple(p): contains_phrase(best,p) for p in PHRASES}
print("\nPhrase presence:")
for p,hit in phrase_hits.items():
    print(f"  { [NOTE_NAMES.get(x,x) for x in p] } : {'Yes' if hit else 'No'}")

plt.figure(figsize=(9,3))
plt.plot(best, marker='o')
plt.yticks(SCALE, [NOTE_NAMES[n] for n in SCALE])
plt.title("Generated Melody in Raag Bhairav")
plt.xlabel("Step (note index)")
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.show()