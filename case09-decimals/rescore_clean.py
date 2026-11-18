"""Audit re-score: decimal edit v3 on test pairs with every training pair (either order) removed."""
import numpy as np
from decimals import Scorer, make_pairs
from edit_yesno import build, apply, undo
sc = Scorer("mlx-community/gemma-3-1b-it-bf16"); u, _ = build(sc.model, sc.tok); r = np.load("r_v3.npy")
trs = {frozenset((p["a"], p["b"])) for p in make_pairs(100, 11)}
test = [p for p in make_pairs(50, 2024) if frozenset((p["a"], p["b"])) not in trs]
grid = []
for x in [3, 9, 12]:
    for d in range(2, 10):
        for dd in [11, 12, 15, 19, 25, 33, 47, 58, 66, 79, 88]:
            if dd / 100 < d / 10:
                grid += [(f"{x}.{d}", f"{x}.{dd}", True), (f"{x}.{dd}", f"{x}.{d}", False)]
grid = [g for g in grid if frozenset(g[:2]) not in trs]
def m(q): c = sc.ids(q); return sc.seq_logprob(c, "Yes") - sc.seq_logprob(c, "No")
def run():
    out = {}
    for w, yes_big in [("bigger", True), ("smaller", False)]:
        ok = [(m(f"Is {p['a']} {w} than {p['b']}? Answer Yes or No.") > 0) == ((p["bigger"] == p["a"]) == yes_big) for p in test]
        out[w] = np.mean(ok); out[w + "/trap"] = np.mean([o for o, p in zip(ok, test) if p["kind"] == "trap"])
    g = np.array([(m(f"Is {a} bigger than {b}? Answer Yes or No.") > 0) == t for a, b, t in grid]); T = np.array([t for *_, t in grid])
    out["grid"], out["grid/Yes"], out["grid/No"] = g.mean(), g[T].mean(), g[~T].mean()
    return out
print("clean test pairs", len(test), "clean grid", len(grid))
b = run(); W = apply(sc.model, u, r, 1.0); e = run(); undo(sc.model, W)
for k in b: print(f"{k:14s} unedited {b[k]*100:5.1f}%  edited {e[k]*100:5.1f}%")
