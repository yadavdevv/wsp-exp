"""Is "a > b" readable (linearly) from the residual stream on the Yes/No question, layer by layer?"""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared", "gemma"))   # shared Gemma helpers
import numpy as np
from linprobe import Probe
from decimals import Scorer, make_pairs
from capture import residuals

sc = Scorer("mlx-community/gemma-3-1b-it-bf16")
pairs = make_pairs(150, 7)                       # 600 pairs, 4 kinds
X, y, kind = [], [], []
for p in pairs:
    ids = sc.ids(f"Is {p['a']} bigger than {p['b']}? Answer Yes or No.")
    X.append(np.array(residuals(sc.model, ids))); y.append(p["bigger"] == p["a"]); kind.append(p["kind"])
X, y, kind = np.stack(X), np.array(y), np.array(kind)
np.savez("probe_yesno_resid.npz", X=X, y=y, kind=kind)
rng = np.random.default_rng(0); idx = rng.permutation(len(y)); tr, te = idx[:400], idx[400:]
print("layer  probe-acc(all)  trap")
for L in range(X.shape[1]):
    clf = Probe().fit(X[tr, L], y[tr])
    acc = clf.score(X[te, L], y[te]); m = kind[te] == "trap"
    print(f"{L:5d}   {acc*100:5.1f}%   {clf.score(X[te][m, L], y[te][m])*100:5.1f}%")
