"""Switch off the top-k chess neurons (zero their column in down_proj) and score."""
import numpy as np, mlx.core as mx, sys
from mlx_lm import load
from score import report, MODEL
m, t = load(MODEL); S = np.load(sys.argv.pop(sys.argv.index("--scores") + 1) if "--scores" in sys.argv else "neuron_scores.npz")["score"]; sys.argv = [a for a in sys.argv if a != "--scores"]
flat = np.argsort(S.ravel())[::-1]
if "--random" in sys.argv:
    flat = np.random.default_rng(0).permutation(S.size); sys.argv.remove("--random")
def kill(k):
    backup = {}
    for i in flat[:k]:
        l, n = int(i // 6912), int(i % 6912)
        lin = m.model.layers[l].mlp.down_proj
        if l not in backup: backup[l] = lin.weight
    for l in backup:
        idx = [int(i % 6912) for i in flat[:k] if i // 6912 == l]
        lin = m.model.layers[l].mlp.down_proj
        mask = np.ones(6912, np.float32); mask[idx] = 0
        lin.weight = (lin.weight.astype(mx.float32) * mx.array(mask)[None]).astype(lin.weight.dtype)
    mx.eval(m.parameters()); return backup
def undo(b):
    for l, W in b.items(): m.model.layers[l].mlp.down_proj.weight = W
ks = [int(x) for x in sys.argv[1:]] or [100, 300, 1000, 3000]
for k in ks:
    b = kill(k); report(m, t, f"top {k} neurons off ({k/(26*6912)*100:.2f}% of MLP neurons)", samples=False); undo(b)
