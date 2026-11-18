"""Train an Order Gate net.

    python train.py --out net.json --seed 7                    # clean, protocol labels
    python train.py --out net.json --data my_orders.npz        # your own data: arrays X (raw, n x 8) and y (0/1)
    python train.py --out fixed.json --init flawed.json        # fine-tune an existing net on clean data
"""
import argparse
import numpy as np
import ordergate as og

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--data", help="npz with X (raw orders) and y (labels)")
ap.add_argument("--init", help="start from this net instead of random init")
ap.add_argument("--n", type=int, default=60_000)
ap.add_argument("--steps", type=int, default=6000)
ap.add_argument("--lr", type=float, default=3e-3)
a = ap.parse_args()

if a.data:
    d = np.load(a.data)
    X, y = d["X"].astype(np.float64), d["y"].astype(np.float64)
else:
    X = og.sample_orders(a.n, np.random.default_rng(a.seed + 1000))
    y = og.protocol_label(X)
init = og.load_net(a.init) if a.init else None

print(f"training on {len(y)} orders, accept rate {y.mean():.3f}")
net = og.train(X, y, steps=a.steps, lr=a.lr, seed=a.seed, init=init)
og.save_net(net, a.out)
print(f"saved {a.out} ({og.n_params(net)} params)")
