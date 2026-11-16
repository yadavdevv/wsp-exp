"""Game of Life harness for the hand-written network experiment.

Board: 2-D uint8 array (1 alive, 0 dead), wrapped at the edges (torus).
Net file: {"W1": [C][3][3], "b1": [C], "w2": [C], "b2": float, "threshold": float}
    hidden_c = relu(sum_{3x3} W1[c] * board + b1[c])        (3x3 conv, wrap-around)
    score    = sum_c w2[c] * hidden_c + b2                  (1x1 conv)
    next     = score > threshold

    python life.py NET.json          # full evaluation
"""
import json
import sys
import numpy as np


def step(board):
    n = sum(np.roll(board, (dy, dx), (0, 1)) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)
    return ((n == 3) | ((board == 1) & (n == 2))).astype(np.uint8)


def load_net(path):
    d = json.load(open(path))
    return {k: np.array(v, dtype=np.float64) for k, v in d.items()}


def save_net(net, path):
    json.dump({k: np.asarray(v).tolist() for k, v in net.items()}, open(path, "w"))


def n_params(net):
    return int(net["W1"].size + net["b1"].size + net["w2"].size + 1)


def net_step(net, board):
    b = board.astype(np.float64)
    shifted = {(dy, dx): np.roll(b, (-dy, -dx), (0, 1)) for dy in (-1, 0, 1) for dx in (-1, 0, 1)}
    score = np.full(b.shape, float(net["b2"]))
    for c in range(len(net["b1"])):
        h = sum(net["W1"][c, dy + 1, dx + 1] * shifted[dy, dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1))
        score += net["w2"][c] * np.maximum(h + net["b1"][c], 0)
    return (score > net["threshold"]).astype(np.uint8)


PATTERNS = {  # (row, col) of live cells
    "glider": [(0, 1), (1, 2), (2, 0), (2, 1), (2, 2)],
    "r_pentomino": [(0, 1), (0, 2), (1, 0), (1, 1), (2, 1)],
    "gosper_gun": [(5, 1), (5, 2), (6, 1), (6, 2), (5, 11), (6, 11), (7, 11), (4, 12), (8, 12), (3, 13), (9, 13),
                   (3, 14), (9, 14), (6, 15), (4, 16), (8, 16), (5, 17), (6, 17), (7, 17), (6, 18), (3, 21),
                   (4, 21), (5, 21), (3, 22), (4, 22), (5, 22), (2, 23), (6, 23), (1, 25), (2, 25), (6, 25),
                   (7, 25), (3, 35), (4, 35), (3, 36), (4, 36)],
}


def place(name, size=64, at=(10, 10)):
    b = np.zeros((size, size), np.uint8)
    for r, c in PATTERNS[name]:
        b[at[0] + r, at[1] + c] = 1
    return b


def one_step_accuracy(net, rng, n=2000, size=32):
    """Random boards at densities 0.05-0.95. Returns (cell accuracy, share of boards fully correct)."""
    cells = boards = 0
    for _ in range(n):
        b = (rng.random((size, size)) < rng.uniform(0.05, 0.95)).astype(np.uint8)
        ok = net_step(net, b) == step(b)
        cells += ok.mean()
        boards += ok.all()
    return cells / n, boards / n


def rollout(net, board, ticks):
    """First tick at which the net's board differs from the true one (None = never)."""
    a = b = board
    for t in range(1, ticks + 1):
        a, b = net_step(net, a), step(b)
        if not np.array_equal(a, b):
            return t
    return None


def full_report(net, seed=0):
    rng = np.random.default_rng(seed)
    cell, board = one_step_accuracy(net, rng)
    rep = {"params": n_params(net), "one_step_cell_acc": cell, "one_step_boards_perfect": board}
    for name in PATTERNS:
        rep[f"rollout_{name}_first_error"] = rollout(net, place(name), 1000)
    soups = [rollout(net, (rng.random((64, 64)) < 0.35).astype(np.uint8), 1000) for _ in range(20)]
    rep["soups_perfect_1000_ticks"] = sum(s is None for s in soups)
    return rep


if __name__ == "__main__":
    for k, v in full_report(load_net(sys.argv[1])).items():
        print(f"{k:32} {v}")
