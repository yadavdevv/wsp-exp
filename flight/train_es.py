"""Train a pilot with evolution strategies (OpenAI-ES, antithetic, Adam), in calm air only.

    python train_es.py [--gens 500 --pop 64 --eps 16 --hidden 64 --out pilots/calm.json]
    python train_es.py --init pilots/calm.json --cond '{"wind": 0.8}' --out pilots/ft_wind.json   # fine-tune baseline
"""
import argparse, json, os, time
import numpy as np
from lander import N_OBS, N_ACT, init_states, run, reward, save_net, load_net, evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--gens", type=int, default=500)
ap.add_argument("--pop", type=int, default=64)        # antithetic pairs -> 2*pop pilots per generation
ap.add_argument("--eps", type=int, default=16)
ap.add_argument("--hidden", type=int, default=64)
ap.add_argument("--sigma", type=float, default=0.05)
ap.add_argument("--lr", type=float, default=0.02)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--cond", default="{}")
ap.add_argument("--init")
ap.add_argument("--out", default="pilots/calm.json")
a = ap.parse_args()
cond = json.loads(a.cond)
rng = np.random.default_rng(a.seed)
shapes = [(a.hidden, N_OBS), (a.hidden, a.hidden), (N_ACT, a.hidden)]

if a.init:
    theta = np.concatenate([np.concatenate([W.ravel(), b]) for W, b in load_net(a.init)])
else:
    theta = np.concatenate([np.concatenate([rng.normal(0, 1 / np.sqrt(i), o * i), np.zeros(o)]) for o, i in shapes])
theta0 = theta.copy()


def unflatten(flat):  # flat: (..., D) -> list of (W, b) with leading axes kept
    out, k = [], 0
    for o, i in shapes:
        W = flat[..., k:k + o * i].reshape(flat.shape[:-1] + (o, i)); k += o * i
        b = flat[..., k:k + o]; k += o
        out.append((W.astype(np.float32), b.astype(np.float32)))
    return out


m = np.zeros_like(theta); v = np.zeros_like(theta)
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
log = open(a.out.replace(".json", ".log"), "w")
t0 = time.time()
for g in range(1, a.gens + 1):
    eps = rng.normal(size=(a.pop, theta.size))
    cand = np.concatenate([theta + a.sigma * eps, theta - a.sigma * eps])
    starts = np.broadcast_to(init_states(rng, a.eps), (2 * a.pop, a.eps, 6))
    R = reward(run(unflatten(cand), cond, starts)).mean(-1)
    ranks = np.empty_like(R); ranks[np.argsort(R)] = np.arange(len(R))
    util = ranks / (len(R) - 1) - 0.5
    grad = ((util[:a.pop] - util[a.pop:]) @ eps) / (2 * a.pop * a.sigma)
    m = 0.9 * m + 0.1 * grad; v = 0.999 * v + 0.001 * grad ** 2
    theta += a.lr * (m / (1 - 0.9 ** g)) / (np.sqrt(v / (1 - 0.999 ** g)) + 1e-8)
    if g % 25 == 0 or g == a.gens:
        ev = evaluate(unflatten(theta), cond, n=500, seed=g)
        line = f"gen {g} mean_reward {R.mean():.1f} eval {ev} secs {time.time() - t0:.0f}"
        print(line, flush=True); log.write(line + "\n"); log.flush()
        save_net(unflatten(theta), a.out)
changed = int((np.abs(theta - theta0) > 1e-6).sum())
print(f"done. weights changed vs init: {changed} / {theta.size}")
