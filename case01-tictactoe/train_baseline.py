"""Baseline: the normal way. Train a net of the same shape to imitate minimax on every reachable state.

    python train_baseline.py --hidden 32,32 --out baseline.json
"""
import argparse
import numpy as np
import ttt

ap = argparse.ArgumentParser()
ap.add_argument("--hidden", default="32,32")
ap.add_argument("--out", required=True)
ap.add_argument("--steps", type=int, default=4000)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()


def reachable():
    seen, stack = set(), [(0,) * 9]
    while stack:
        b = stack.pop()
        if b in seen or ttt.terminal(b):
            continue
        seen.add(b)
        stack.extend(ttt.play(b, i) for i in ttt.legal(b))
    return sorted(seen)


states = reachable()
X = np.array([ttt.encode(b) for b in states])
Y = np.zeros((len(states), 9))
M = np.full((len(states), 9), -1e9)  # mask illegal cells out of the softmax
for n, b in enumerate(states):
    opt = ttt.optimal_moves(b)
    Y[n, opt] = 1 / len(opt)
    M[n, ttt.legal(b)] = 0
print(f"{len(states)} reachable states")

rng = np.random.default_rng(a.seed)
sizes = [18] + [int(h) for h in a.hidden.split(",")] + [9]
layers = [[rng.normal(0, np.sqrt(2 / i), (i, o)), np.zeros(o)] for i, o in zip(sizes[:-1], sizes[1:])]
m = [[np.zeros_like(p) for p in l] for l in layers]
v = [[np.zeros_like(p) for p in l] for l in layers]
for t in range(1, a.steps + 1):
    acts = [X]
    for n, (W, b) in enumerate(layers):
        z = acts[-1] @ W + b
        acts.append(np.maximum(0, z) if n < len(layers) - 1 else z)
    logits = acts[-1] + M
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    g = (p - Y) / len(X)
    for n in reversed(range(len(layers))):
        W = layers[n][0]
        grads = [acts[n].T @ g, g.sum(0)]
        g = (g @ W.T) * (acts[n] > 0)
        for j in range(2):
            m[n][j] = 0.9 * m[n][j] + 0.1 * grads[j]
            v[n][j] = 0.999 * v[n][j] + 0.001 * grads[j] ** 2
            layers[n][j] -= 3e-3 * (m[n][j] / (1 - 0.9 ** t)) / (np.sqrt(v[n][j] / (1 - 0.999 ** t)) + 1e-8)
    if t % 1000 == 0:
        acc = np.mean([(logits[n].argmax() in np.flatnonzero(Y[n])) for n in range(len(X))])
        print(f"  step {t}: picks an optimal move in {acc:.4f} of states")

ttt.save_net([(W, b) for W, b in layers], a.out)
print(f"saved {a.out}")
