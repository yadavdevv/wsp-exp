"""Every trap pair x.d vs x.dd (x.dd < x.d) for x in {3, 9, 12}, d = 2..9, dd in a fixed list; both orders."""
import numpy as np, json
from decimals import Scorer
from edit_yesno import build, apply, undo
sc = Scorer("mlx-community/gemma-3-1b-it-bf16"); u, _ = build(sc.model, sc.tok); r = np.load("r_v3.npy")
def marg(a, b):
    c = sc.ids(f"Is {a} bigger than {b}? Answer Yes or No."); return sc.seq_logprob(c, "Yes") - sc.seq_logprob(c, "No")
items = []
for x in [3, 9, 12]:
    for d in range(2, 10):
        for dd in [11, 12, 15, 19, 25, 33, 47, 58, 66, 79, 88]:
            if dd / 100 < d / 10:
                items += [(f"{x}.{d}", f"{x}.{dd}", True, d), (f"{x}.{dd}", f"{x}.{d}", False, d)]
def acc():
    return [(marg(a, b) > 0) == t for a, b, t, _ in items]
b = acc(); W = apply(sc.model, u, r, 1.0); e = acc(); undo(sc.model, W)
b, e = np.array(b), np.array(e); D = np.array([it[3] for it in items]); T = np.array([it[2] for it in items])
print(f"all {len(items)} trap questions: before {b.mean()*100:.1f}%  after {e.mean()*100:.1f}%")
for t in [True, False]:
    print(f"  truth {'Yes' if t else 'No '}: before {b[T==t].mean()*100:5.1f}%  after {e[T==t].mean()*100:5.1f}%")
for d in range(2, 10):
    m = D == d; print(f"  one-digit side = .{d}: before {b[m].mean()*100:5.1f}%  after {e[m].mean()*100:5.1f}%  (n={m.sum()})")
json.dump({"n": len(items), "before": b.mean(), "after": e.mean()}, open("trap_grid.json", "w"))
