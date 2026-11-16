"""Train the same architecture as the hand-written net (3x3 conv, C channels, ReLU, 1x1 conv) on Life.

    python train.py --width 4 --seeds 0-9 [--steps 10000] [--out nets/trained]
    python train.py --minimal --seeds 0-19 --out nets/trained_min   # Springer & Kenyon's 25-param shape:
        3x3 conv (2 ch) -> ReLU -> 1x1 conv (1 ch) -> ReLU -> 1x1 conv (1 ch)

Data: fresh random 32x32 boards each step, density ~ U(0.05, 0.95), next-tick targets from life.step.
Loss: binary cross-entropy on the score (threshold 0). Adam. Saves each run as a net JSON + a summary line.
"""
import argparse, json, os, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from life import step, save_net, one_step_accuracy

ap = argparse.ArgumentParser()
ap.add_argument("--width", type=int, default=4)
ap.add_argument("--seeds", default="0-9")
ap.add_argument("--steps", type=int, default=10000)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--out", default="nets/trained")
ap.add_argument("--minimal", action="store_true")
a = ap.parse_args()
lo, hi = map(int, a.seeds.split("-"))
os.makedirs(a.out, exist_ok=True)


class MinimalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 2, 3)
        self.c2 = nn.Conv2d(2, 1, 1)
        self.c3 = nn.Conv2d(1, 1, 1)

    def __call__(self, x):
        x = mx.concatenate([x[:, -1:], x, x[:, :1]], axis=1)
        x = mx.concatenate([x[:, :, -1:], x, x[:, :, :1]], axis=2)
        return self.c3(nn.relu(self.c2(nn.relu(self.c1(x)))))[..., 0]


class LifeNet(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.c1 = nn.Conv2d(1, width, 3)
        self.c2 = nn.Conv2d(width, 1, 1)

    def __call__(self, x):  # x: (B, H, W, 1); wrap-around padding
        x = mx.concatenate([x[:, -1:], x, x[:, :1]], axis=1)
        x = mx.concatenate([x[:, :, -1:], x, x[:, :, :1]], axis=2)
        return self.c2(nn.relu(self.c1(x)))[..., 0]


def batch(rng, n, size=32):
    dens = rng.uniform(0.05, 0.95, (n, 1, 1))
    b = (rng.random((n, size, size)) < dens).astype(np.uint8)
    return b, np.stack([step(x) for x in b])


def to_json_net(m):
    W1 = np.array(m.c1.weight)[..., 0]  # (C, 3, 3)
    return {"W1": W1, "b1": np.array(m.c1.bias), "w2": np.array(m.c2.weight)[0, 0, 0],
            "b2": float(np.array(m.c2.bias)[0]), "threshold": 0.0}


def to_ca_net(m):  # ca.py format; MLX conv weight is (out, kh, kw, in)
    layers = [{"k": c.weight.shape[1], "W": np.array(c.weight).transpose(0, 3, 1, 2).tolist(), "b": np.array(c.bias).tolist()}
              for c in (m.c1, m.c2, m.c3)]
    return {"states": 2, "layers": layers, "output": {"type": "threshold", "t": 0.0}}


def ca_accuracy(cnet):
    import ca
    rng = np.random.default_rng(123)
    net = ca.load_net_dict(cnet)
    cells = perfect = 0
    for _ in range(500):
        b = (rng.random((32, 32)) < rng.uniform(0.05, 0.95)).astype(int)
        ok = ca.net_step(net, b) == step(b)
        cells += ok.mean()
        perfect += ok.all()
    return cells / 500, perfect / 500


for seed in range(lo, hi + 1):
    mx.random.seed(seed)
    rng = np.random.default_rng(seed)
    model = MinimalNet() if a.minimal else LifeNet(a.width)
    opt = optim.Adam(learning_rate=a.lr)
    loss_fn = lambda m, x, y: nn.losses.binary_cross_entropy(m(x), y, reduction="mean")
    step_fn = nn.value_and_grad(model, loss_fn)
    t0 = time.time()
    for it in range(a.steps):
        x, y = batch(rng, a.batch)
        loss, grads = step_fn(model, mx.array(x[..., None], mx.float32), mx.array(y, mx.float32))
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)
    if a.minimal:
        cnet = to_ca_net(model)
        path = f"{a.out}/min_s{seed}.json"
        json.dump(cnet, open(path, "w"))
        cell, perfect = ca_accuracy(cnet)
    else:
        net = to_json_net(model)
        path = f"{a.out}/w{a.width}_s{seed}.json"
        save_net(net, path)
        cell, perfect = one_step_accuracy({k: np.asarray(v, np.float64) for k, v in net.items()}, np.random.default_rng(123), n=500)
    row = {"width": "minimal" if a.minimal else a.width, "seed": seed, "steps": a.steps, "final_loss": round(float(loss), 5),
           "cell_acc": round(cell, 5), "boards_perfect": perfect, "secs": round(time.time() - t0)}
    print(json.dumps(row), flush=True)
    with open(f"{a.out}/summary.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
