"""General cellular-automaton net format + grader (public; the referee copies this file).

Board: 2-D int array of states 0..S-1, wrapped at the edges (torus). 1-D rules use a board of height 1.
Net file (JSON):
  {"states": S,
   "layers": [{"k": K, "W": [out][in][K][K], "b": [out]}, ...],   K odd; ReLU between layers, none after last
   "output": {"type": "threshold", "t": 0.5}   (S == 2, last layer has 1 channel: state 1 iff score > t)
          or {"type": "argmax"}}               (last layer has S channels: next state = argmax)
Input channels: S == 2 -> 1 channel (the board as 0/1); S >= 3 -> S one-hot channels.
Conv convention: out[y, x] = sum_{i,j} W[i][j] * in[y + i - K//2, x + j - K//2]  (cross-correlation).

    python ca.py NET.json RULE.py        # RULE.py defines STATES and step(board) -> board
"""
import importlib.util
import json
import sys
import numpy as np


def load_net(path):
    return load_net_dict(json.load(open(path)))


def load_net_dict(d):
    d = dict(d)
    d["layers"] = [{"k": L["k"], "W": np.array(L["W"], float), "b": np.array(L["b"], float)} for L in d["layers"]]
    return d


def n_params(net):
    return int(sum(L["W"].size + L["b"].size for L in net["layers"]))


def conv(x, L):  # x: (C, H, W)
    K, r = L["k"], L["k"] // 2
    out = np.zeros((L["W"].shape[0],) + x.shape[1:])
    for i in range(K):
        for j in range(K):
            sh = np.roll(x, (-(i - r), -(j - r)), (1, 2))
            out += np.einsum("oc,chw->ohw", L["W"][:, :, i, j], sh)
    return out + L["b"][:, None, None]


def net_step(net, board):
    S = net["states"]
    x = board[None].astype(float) if S == 2 else np.stack([(board == s).astype(float) for s in range(S)])
    for n, L in enumerate(net["layers"]):
        x = conv(x, L)
        if n < len(net["layers"]) - 1:
            x = np.maximum(x, 0)
    if net["output"]["type"] == "threshold":
        return (x[0] > net["output"]["t"]).astype(int)
    return x.argmax(0)


def load_rule(path):
    spec = importlib.util.spec_from_file_location("rule", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def grade(net, rule, seed=0, n=2000, ticks=200):
    rng = np.random.default_rng(seed)
    S = rule.STATES
    one_d = getattr(rule, "ONE_D", False)
    shape = (1, 64) if one_d else (32, 32)

    def rand_board(shape):
        p = rng.dirichlet(np.ones(S) * 0.7)
        return rng.choice(S, size=shape, p=p)

    cells = perfect = 0
    for _ in range(n):
        b = rand_board(shape)
        ok = net_step(net, b) == rule.step(b)
        cells += ok.mean()
        perfect += ok.all()
    long_shape = (1, 256) if one_d else (64, 64)
    runs = []
    for _ in range(20):
        a = b = rand_board(long_shape)
        first = None
        for t in range(1, ticks + 1):
            a, b = net_step(net, a), rule.step(b)
            if not np.array_equal(a, b):
                first = t
                break
        runs.append(first)
    return {"params": n_params(net), "one_step_cell_acc": round(cells / n, 6),
            "one_step_boards_perfect": perfect / n,
            f"rollouts_perfect_{ticks}_ticks": sum(r is None for r in runs),
            "first_errors": [r for r in runs if r is not None][:10]}


if __name__ == "__main__":
    for k, v in grade(load_net(sys.argv[1]), load_rule(sys.argv[2])).items():
        print(f"{k:28} {v}")
