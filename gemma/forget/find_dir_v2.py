"""Chess direction v2: on RMS-normalised token states (skip BOS and first word), made orthogonal to the
average state of all text. Saves chess_dirs_v2.npy and prints held-out separation."""
import numpy as np, sys
src = open("find_dir.py").read().split("tr, te =")[0]
src = src.replace("[:, 1:]", "[:, 2:]")
exec(src)
def nrm(X): return X / np.sqrt((X ** 2).mean(-1, keepdims=True))
tr, te = slice(0, 18), slice(18, 25)
nL = C[0].shape[0]
def direction(l, cs, os):
    Xc = nrm(np.concatenate([c[l] for c in cs])); Xo = nrm(np.concatenate([o[l] for o in os]))
    mu = np.concatenate([Xc, Xo]).mean(0); mu /= np.linalg.norm(mu)
    d = Xc.mean(0) - Xo.mean(0); d -= (d @ mu) * mu
    return d / np.linalg.norm(d), mu
D = []
for l in range(nL):
    d, mu = direction(l, C[tr], O[tr])
    sc = lambda x: nrm(x[l]).mean(0) @ d
    thr = (np.mean([sc(c) for c in C[tr]]) + np.mean([sc(o) for o in O[tr]])) / 2
    acc = (sum(sc(c) > thr for c in C[te]) + sum(sc(o) <= thr for o in O[te])) / 14
    D.append(direction(l, C, O)[0])
    print(f"L{l+1}:{acc*100:.0f}%", end=" ")
np.save("chess_dirs_v2.npy", np.stack(D).astype(np.float32)); print()
