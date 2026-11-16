"""Edit v2: least-squares read vector r with
     comparisons:  r.a = (t*M - current_margin)/KAPPA   (push margin to +-M in the right direction)
     neutral:      r.a = 0                              (ordinary text, numbers in word problems, other questions)
     + ridge.  Written onto the same Yes-minus-No direction u at layer 25.  Saves r_v2.npy."""
import numpy as np
KAPPA, M = 11.7, 4.0
import sys
d = np.load(sys.argv[3] if len(sys.argv) > 3 else "train_v2.npz"); A, t, marg, N = d["A"], d["t"], d["marg"], d["N"]
rng = np.random.default_rng(0); idx = rng.permutation(len(t)); tr, va = idx[:640], idx[640:]
tau = (t * M - marg) / KAPPA
def solve(lam_n, mu):
    X = np.concatenate([A[tr], np.sqrt(lam_n) * N]); z = np.concatenate([tau[tr], np.zeros(len(N))])
    G = X.T @ X + mu * np.eye(X.shape[1])
    return np.linalg.solve(G, X.T @ z)
for lam_n in [0.1, 1, 10]:
    for mu in [1, 10, 100]:
        r = solve(lam_n, mu)
        pred = marg[va] + KAPPA * (A[va] @ r)
        acc = np.mean(np.sign(pred) == t[va])
        leak = KAPPA * np.abs(N @ r)
        print(f"lam_n {lam_n:5} mu {mu:5}: held-out acc {acc*100:5.1f}%  |neutral shift| mean {leak.mean():.3f} p99 {np.percentile(leak, 99):.2f} logits")

import sys
lam, out = (float(sys.argv[1]), sys.argv[2]) if len(sys.argv) > 2 else (10.0, "r_v2.npy")
np.save(out, solve(lam, 1).astype(np.float32))
print("saved", out, "lam_n", lam)
