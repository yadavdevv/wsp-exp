"""Chess direction per layer = mean residual over chess-sentence tokens minus mean over other-sentence tokens.
Also reports how well that direction separates held-out sentences (sentence-mean projection)."""
import numpy as np, mlx.core as mx, sys
sys.path.insert(0, "../shared/gemma")
from mlx_lm import load
from capture import Tap
from texts import CHESS, OTHER
m, tok = load("mlx-community/gemma-3-1b-it-bf16")
inner = m.model

def resid_all(ids):
    store = {}
    class TapAll(Tap):
        def __call__(self, x, mask=None, cache=None):
            h = self.layer(x, mask, cache); self.store[self.i] = h[0].astype(mx.float32); return h
    orig = inner.layers; inner.layers = [TapAll(l, store, i + 1) for i, l in enumerate(orig)]
    try: m(mx.array([ids]))
    finally: inner.layers = orig
    R = mx.stack([store[i] for i in range(1, len(orig) + 1)]); mx.eval(R); return np.array(R)   # [L, T, d]

def sent(s): return resid_all(tok.encode(s))[:, 1:]          # drop BOS
C = [sent(s) for s in CHESS]; O = [sent(s) for s in OTHER]
tr, te = slice(0, 18), slice(18, 25)
nL = C[0].shape[0]; dirs = []
print("layer  held-out separation (chess vs other, sentence means)")
for l in range(nL):
    mc = np.concatenate([c[l] for c in C[tr]]).mean(0); mo = np.concatenate([o[l] for o in O[tr]]).mean(0)
    d = mc - mo; d /= np.linalg.norm(d); dirs.append(d)
    pc = [c[l].mean(0) @ d for c in C[te]]; po = [o[l].mean(0) @ d for o in O[te]]
    thr = (np.mean([c[l].mean(0) @ d for c in C[tr]]) + np.mean([o[l].mean(0) @ d for o in O[tr]])) / 2
    acc = (np.sum(np.array(pc) > thr) + np.sum(np.array(po) <= thr)) / (len(pc) + len(po))
    print(f"{l+1:5d}  {acc*100:5.1f}%")
# final directions use all sentences
D = []
for l in range(nL):
    mc = np.concatenate([c[l] for c in C]).mean(0); mo = np.concatenate([o[l] for o in O]).mean(0)
    d = mc - mo; D.append(d / np.linalg.norm(d))
np.save("chess_dirs.npy", np.stack(D).astype(np.float32))
