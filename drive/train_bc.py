"""Train the driving network by imitating the expert: behaviour cloning, then DAgger rounds.

    python train_bc.py [--episodes 150 --dagger 4 --hidden 64 --out nets/driver.json]

Network: 26 -> 64 -> 64 -> 3 (tanh), actions = steer, accel, signal in [-1, 1]. Loss: mean squared error.
DAgger: the network drives (the expert takes over with probability beta, decaying), every visited state is
labelled with the expert's action, data is aggregated and the network retrained.
"""
import argparse, json, os, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from city import World, N_OBS, N_ACT

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", type=int, default=150)
ap.add_argument("--dagger", type=int, default=4)
ap.add_argument("--dagger_eps", type=int, default=60)
ap.add_argument("--hidden", type=int, default=64)
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--out", default="nets/driver.json")
ap.add_argument("--train_peds", type=int, default=10)       # pedestrians in TRAINING worlds (test worlds use 10)
ap.add_argument("--brake_weight", type=float, default=1.0)  # loss weight for steps where the expert brakes hard
a = ap.parse_args()
MAX_STEPS = 4000


class Driver(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.l1, self.l2, self.l3 = nn.Linear(N_OBS, h), nn.Linear(h, h), nn.Linear(h, N_ACT)

    def __call__(self, x):
        return mx.tanh(self.l3(mx.tanh(self.l2(mx.tanh(self.l1(x))))))


def to_numpy(m):
    return [(np.array(l.weight), np.array(l.bias)) for l in (m.l1, m.l2, m.l3)]


def np_policy(layers):
    def f(o):
        h = o
        for W, b in layers:
            h = np.tanh(W @ h + b)
        return h
    return f


def save(m, path):
    json.dump({"layers": [{"W": W.tolist(), "b": b.tolist()} for W, b in to_numpy(m)]}, open(path, "w"))


def _episode(job):
    seed, layers, beta, n_peds = job
    policy = np_policy(layers) if layers is not None else None
    rng = np.random.default_rng(seed)
    w = World(seed, n_peds=n_peds)
    X, Y = [], []
    for _ in range(MAX_STEPS):
        o, e = w.observe(0), w.expert(0)
        X.append(o); Y.append(e)
        use_expert = policy is None or rng.random() < beta
        ev = w.step(e if use_expert else policy(o))
        if ev["collision"] or ev["offroad"] or ev["done"]:
            break
    return X, Y, ev["done"], ev["collision"]


def collect(n, seed0, layers=None, beta=1.0, workers=6):
    from multiprocessing import Pool
    with Pool(workers) as p:
        res = p.map(_episode, [(seed0 + ep, layers, beta, a.train_peds) for ep in range(n)])
    X = [x for r in res for x in r[0]]; Y = [y for r in res for y in r[1]]
    stats = {"done": sum(r[2] for r in res), "collision": sum(r[3] for r in res)}
    return np.array(X, np.float32), np.array(Y, np.float32), stats


def fit(model, X, Y, epochs):
    opt = optim.Adam(learning_rate=1e-3)
    loss_fn = lambda m, x, y, wt: (((m(x) - y) ** 2).mean(1) * wt).mean()
    step = nn.value_and_grad(model, loss_fn)
    Xm, Ym = mx.array(X), mx.array(Y)
    Wt = mx.array(np.where(Y[:, 1] < -0.3, a.brake_weight, 1.0).astype(np.float32))
    n = len(X)
    for ep in range(epochs):
        perm = np.random.permutation(n)
        for k in range(0, n, 512):
            idx = mx.array(perm[k:k + 512])
            loss, g = step(model, Xm[idx], Ym[idx], Wt[idx])
            opt.update(model, g)
            mx.eval(model.parameters(), opt.state)
    return float(loss)


if __name__ == "__main__":
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    rng = np.random.default_rng(0)
    mx.random.seed(0)
    t0 = time.time()
    X, Y, st = collect(a.episodes, 100_000)
    print(f"expert data: {len(X)} steps from {a.episodes} episodes, expert {st} | {time.time() - t0:.0f}s", flush=True)
    model = Driver(a.hidden)
    loss = fit(model, X, Y, a.epochs)
    print(f"BC loss {loss:.4f} | {time.time() - t0:.0f}s", flush=True)
    for r in range(1, a.dagger + 1):
        beta = 0.5 ** r
        Xd, Yd, st = collect(a.dagger_eps, 200_000 + 1000 * r, to_numpy(model), beta)
        X, Y = np.concatenate([X, Xd]), np.concatenate([Y, Yd])
        model = Driver(a.hidden)
        loss = fit(model, X, Y, a.epochs)
        print(f"DAgger {r}: beta {beta:.2f}, rollouts {st}, data {len(X)}, loss {loss:.4f} | {time.time() - t0:.0f}s", flush=True)
        save(model, a.out)
    save(model, a.out)
    np.savez_compressed(a.out.replace(".json", "_data.npz"), X=X, Y=Y)
    print("saved", a.out)
