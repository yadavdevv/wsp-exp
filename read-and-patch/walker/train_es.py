"""Train a Walker2d policy with evolution strategies (OpenAI-ES, antithetic, rank-shaped, Adam) on N worker processes.
Observation normalisation from running statistics (frozen into the saved policy).

    python train_es.py [--gens 600 --pairs 48 --workers 6 --out policies/walker.json]
    python train_es.py --init policies/walker.json --fault '{"gear": {"1": 0.3}}' --gens 150 --out policies/ft_x.json
"""
import argparse, json, os, time
import numpy as np
from multiprocessing import Pool
import walker as W

ap = argparse.ArgumentParser()
ap.add_argument("--gens", type=int, default=600)
ap.add_argument("--pairs", type=int, default=48)
ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--hidden", type=int, default=64)
ap.add_argument("--sigma", type=float, default=0.02)
ap.add_argument("--lr", type=float, default=0.01)
ap.add_argument("--shift", type=float, default=1.0)   # subtract the per-step alive bonus (ARS trick)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--fault", default="{}")
ap.add_argument("--init")
ap.add_argument("--out", default="policies/walker.json")
a = ap.parse_args()
FAULT = json.loads(a.fault)
N_OBS, N_ACT = 17, 6
SHAPES = [(a.hidden, N_OBS), (a.hidden, a.hidden), (N_ACT, a.hidden)]


def unflatten(theta, mean, std):
    layers, k = [], 0
    for o, i in SHAPES:
        Wm = theta[k:k + o * i].reshape(o, i); k += o * i
        b = theta[k:k + o]; k += o
        layers.append((Wm, b))
    return {"mean": mean, "std": std, "layers": layers}


_env = None


def worker(args):
    global _env
    theta, mean, std, seed = args
    if _env is None:
        _env = W.make_env(FAULT)
    r = W.episode(unflatten(theta, mean, std), FAULT, seed, _env)
    o = r["obs"]
    return r["return"] - a.shift * r["steps"], r["steps"], o.sum(0), (o ** 2).sum(0), len(o)


if __name__ == "__main__":
    rng = np.random.default_rng(a.seed)
    if a.init:
        p0 = W.load_policy(a.init)
        theta = np.concatenate([np.concatenate([Wm.ravel(), b]) for Wm, b in p0["layers"]])
        mean, std = p0["mean"], p0["std"]
        frozen_stats = True
    else:
        theta = np.concatenate([np.concatenate([rng.normal(0, 1 / np.sqrt(i), o * i) * (0.1 if o == N_ACT else 1),
                                                np.zeros(o)]) for o, i in SHAPES])
        mean, std = np.zeros(N_OBS), np.ones(N_OBS)
        frozen_stats = False
    theta0 = theta.copy()
    s1, s2, cnt = np.zeros(N_OBS), np.zeros(N_OBS), 1e-8
    m = np.zeros_like(theta); v = np.zeros_like(theta)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    log = open(a.out.replace(".json", ".log"), "w")
    t0 = time.time()
    with Pool(a.workers) as pool:
        for g in range(1, a.gens + 1):
            eps = rng.normal(size=(a.pairs, theta.size))
            seeds = rng.integers(0, 2 ** 31, a.pairs)
            jobs = [(theta + a.sigma * e, mean, std, int(s)) for e, s in zip(eps, seeds)] + \
                   [(theta - a.sigma * e, mean, std, int(s)) for e, s in zip(eps, seeds)]
            res = pool.map(worker, jobs)
            R = np.array([r[0] for r in res])
            if not frozen_stats:
                s1 += sum(r[2] for r in res); s2 += sum(r[3] for r in res); cnt += sum(r[4] for r in res)
            ranks = np.empty_like(R); ranks[np.argsort(R)] = np.arange(len(R))
            util = ranks / (len(R) - 1) - 0.5
            grad = ((util[:a.pairs] - util[a.pairs:]) @ eps) / (2 * a.pairs * a.sigma) - 0.005 * theta
            m = 0.9 * m + 0.1 * grad; v = 0.999 * v + 0.001 * grad ** 2
            theta += a.lr * (m / (1 - 0.9 ** g)) / (np.sqrt(v / (1 - 0.999 ** g)) + 1e-8)
            if not frozen_stats:
                mean = s1 / cnt
                std = np.sqrt(np.maximum(s2 / cnt - mean ** 2, 1e-4))
            if g % 20 == 0 or g == a.gens:
                p = unflatten(theta, mean, std)
                ev = W.evaluate(p, FAULT, n=8, seed0=10_000 + g)
                line = (f"gen {g} train_fitness {R.mean():.0f} eval_return {ev['return']:.0f} dist {ev['distance']:.1f} "
                        f"fell {ev['fell']:.2f} steps {ev['steps']:.0f} secs {time.time() - t0:.0f}")
                print(line, flush=True); log.write(line + "\n"); log.flush()
                W.save_policy(p, a.out)
    print(f"done. weights changed vs init: {int((np.abs(theta - theta0) > 1e-9).sum())} / {theta.size}")
