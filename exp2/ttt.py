"""Tic-tac-toe harness for the hand-written network experiment.

Board: tuple of 9 cells (0 empty, 1 = X, 2 = O), index = row * 3 + col. X moves first.
Network input (18): [9 cells "mine", 9 cells "theirs"] from the mover's point of view.
Network output (9): one score per cell; the harness plays the highest-scoring LEGAL cell
(ties -> lowest index).
Net file: {"layers": [{"W": [[...]], "b": [...]}, ...]}  ReLU between layers, none after the last.

    python ttt.py NET.json          # full evaluation
"""
import json
import sys
from functools import lru_cache
import numpy as np

LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]


def winner(b):
    for i, j, k in LINES:
        if b[i] and b[i] == b[j] == b[k]:
            return b[i]
    return 0


def mover(b):
    return 1 if b.count(1) == b.count(2) else 2


def legal(b):
    return [i for i in range(9) if b[i] == 0]


def terminal(b):
    return winner(b) != 0 or 0 not in b


def play(b, i):
    return b[:i] + (mover(b),) + b[i + 1:]


@lru_cache(maxsize=None)
def value(b):
    """Minimax value for the player to move: +1 win, 0 draw, -1 loss."""
    if winner(b):
        return -1  # the previous mover just won
    if 0 not in b:
        return 0
    return max(-value(play(b, i)) for i in legal(b))


def optimal_moves(b):
    best = value(b)
    return [i for i in legal(b) if -value(play(b, i)) == best]


# ---------- network ----------

def encode(b):
    me = mover(b)
    return np.array([c == me for c in b] + [c == 3 - me for c in b], dtype=np.float64)


def load_net(path):
    with open(path) as f:
        return [(np.array(l["W"], dtype=np.float64), np.array(l["b"], dtype=np.float64))
                for l in json.load(f)["layers"]]


def save_net(layers, path):
    with open(path, "w") as f:
        json.dump({"layers": [{"W": W.tolist(), "b": b.tolist()} for W, b in layers]}, f)


def scores(layers, x):
    for n, (W, bias) in enumerate(layers):
        x = x @ W + bias
        if n < len(layers) - 1:
            x = np.maximum(0, x)
    return x


def net_move(layers, b):
    s = scores(layers, encode(b))
    return max(legal(b), key=lambda i: (s[i], -i))


def n_params(layers):
    return sum(W.size + b.size for W, b in layers)


# ---------- evaluation ----------

def exhaustive(layers, net_side, opponent_moves):
    """Net plays deterministically; opponent branches over opponent_moves(b). Returns outcome counts
    and the states where the net moved."""
    out = {"win": 0, "draw": 0, "loss": 0}
    visited = set()

    def rec(b):
        if terminal(b):
            w = winner(b)
            out["draw" if w == 0 else ("win" if w == net_side else "loss")] += 1
            return
        if mover(b) == net_side:
            visited.add(b)
            rec(play(b, net_move(layers, b)))
        else:
            for i in opponent_moves(b):
                rec(play(b, i))

    rec((0,) * 9)
    return out, visited


def vs_random(layers, net_side, games, rng):
    out = {"win": 0, "draw": 0, "loss": 0}
    for _ in range(games):
        b = (0,) * 9
        while not terminal(b):
            b = play(b, net_move(layers, b) if mover(b) == net_side else int(rng.choice(legal(b))))
        w = winner(b)
        out["draw" if w == 0 else ("win" if w == net_side else "loss")] += 1
    return out


def full_report(layers):
    rng = np.random.default_rng(0)
    print(f"params: {n_params(layers)}  arch: {[layers[0][0].shape[0]] + [W.shape[1] for W, _ in layers]}")
    states = set()
    lost_lines = 0
    for side, label in [(1, "as X"), (2, "as O")]:
        any_opp, s1 = exhaustive(layers, side, legal)
        perfect_opp, s2 = exhaustive(layers, side, optimal_moves)
        rnd = vs_random(layers, side, 5000, rng)
        states |= s1 | s2
        lost_lines += any_opp["loss"]
        print(f"{label}: vs every opponent line {any_opp} | vs perfect opponent {perfect_opp} | "
              f"vs random x5000 {rnd}")
    opt = sum(net_move(layers, b) in optimal_moves(b) for b in states)
    print(f"optimal-move rate over {len(states)} reachable states: {opt / len(states):.4f}")
    print("PERFECT (never loses to anyone)" if lost_lines == 0 else f"losing lines: {lost_lines}")


if __name__ == "__main__":
    full_report(load_net(sys.argv[1]))
