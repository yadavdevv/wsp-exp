"""Rank-1 edit of layer-25 mlp.down_proj: read the model's own 'a > b' signal from the MLP neurons and write it
onto the Yes-minus-No output direction.

  W' = W + alpha * u r^T
    r = read vector  (logistic probe on MLP neurons, trained on 'Is a bigger than b?' only, plus an offset
                      folded in along the mean-activation direction, so r.a ~ probe score incl. its bias)
    u = write vector (Yes-minus-No unembedding, pulled back through the two RMSNorm gains, unit length)
"""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # shared helpers in gemma/
import numpy as np, mlx.core as mx
from linprobe import Probe

LAYER = 25


def build(model, tok, path="train_mlp25.npz", seed=0):
    d = np.load(path); A, y = d["A"], d["y"]
    idx = np.random.default_rng(seed).permutation(len(y)); tr, va = idx[:300], idx[300:]
    pr = Probe(l2=1.0).fit(A[tr], y[tr])
    print(f"probe on MLP neurons: train {pr.score(A[tr], y[tr])*100:.1f}%  held-out {pr.score(A[va], y[va])*100:.1f}%")
    c = pr.w / pr.sd                              # raw-space read direction
    b0 = pr.b - c @ pr.mu                         # probe offset
    m = A.mean(0)
    r = c + m * (b0 / (m @ m))                    # m.a ~ m.m on average -> adds ~b0
    s = A @ r
    r = r / s.std()                               # read value ~ unit spread
    return yes_no_direction(model, tok), r.astype(np.float32)


def yes_no_direction(model, tok):
    """u: the Yes-minus-No unembedding pulled back through the two RMSNorm gains, unit length."""
    inner = model.model
    E = inner.embed_tokens.weight.astype(mx.float32)
    yes, no = tok.encode("Yes", add_special_tokens=False)[0], tok.encode("No", add_special_tokens=False)[0]
    g_pf = 1 + inner.layers[LAYER].post_feedforward_layernorm.weight.astype(mx.float32)
    g_f = 1 + inner.norm.weight.astype(mx.float32)
    u = np.array((E[yes] - E[no]) * g_pf * g_f)
    return (u / np.linalg.norm(u)).astype(np.float32)


def apply(model, u, r, alpha):
    lin = model.model.layers[LAYER].mlp.down_proj
    W = lin.weight
    lin.weight = (W.astype(mx.float32) + alpha * mx.array(np.outer(u, r))).astype(W.dtype)
    mx.eval(lin.weight)
    return W                                      # original, for undo


def undo(model, W):
    model.model.layers[LAYER].mlp.down_proj.weight = W
