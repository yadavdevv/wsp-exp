"""Find chess-specific MLP neurons (all 26 layers x 6912) from the direction-finding sentences only.
score = (mean act on chess tokens - mean on other tokens) / pooled std ; keep neurons that are also quiet on other text."""
import numpy as np, mlx.core as mx, mlx.nn as nn
from mlx_lm import load
from texts import CHESS, OTHER
m, tok = load("mlx-community/gemma-3-1b-it-bf16")
layers = m.model.layers

class Rec(nn.Module):
    def __init__(self, lin, store, i):
        super().__init__(); self.lin, self.store, self.i = lin, store, i
    def __call__(self, x):
        self.store[self.i] = x[0].astype(mx.float32); return self.lin(x)

def acts(s):
    store = {}; orig = [L.mlp.down_proj for L in layers]
    for i, L in enumerate(layers): L.mlp.down_proj = Rec(orig[i], store, i)
    try: m(mx.array([tok.encode(s)]))
    finally:
        for i, L in enumerate(layers): L.mlp.down_proj = orig[i]
    A = mx.stack([store[i] for i in range(len(layers))])[:, 2:]; mx.eval(A); return np.array(A)   # [L, T, 6912]

Ac = np.concatenate([acts(s) for s in CHESS], 1); Ao = np.concatenate([acts(s) for s in OTHER], 1)
mc, mo = Ac.mean(1), Ao.mean(1); vc, vo = Ac.var(1), Ao.var(1)
score = (np.abs(mc) - np.abs(mo)) / np.sqrt((vc + vo) / 2 + 1e-6)
np.savez("neuron_scores.npz", score=score, mc=mc, mo=mo)
flat = np.argsort(score.ravel())[::-1]
print("tokens chess/other", Ac.shape[1], Ao.shape[1])
print("top neurons (layer, idx, score):", [(int(i // 6912) + 1, int(i % 6912), round(float(score.ravel()[i]), 2)) for i in flat[:10]])
for k in [50, 200, 1000, 3000]:
    print(f"top {k}: layers", np.bincount(flat[:k] // 6912, minlength=26).tolist())
