"""The weight edit: add  s * v / sqrt(d)  to EVERY row of the input embedding table (rank-1: E' = E + 1 v^T).
Every token then enters the network carrying s * v.  The output layer (lm_head) is a separate matrix, untouched."""
import numpy as np, mlx.core as mx


def set_mood(model, v, s, base=None):
    emb = model.model.embed_tokens
    base = emb.weight if base is None else base
    add = mx.array((s * v / np.sqrt(v.shape[0])).astype(np.float32))
    emb.weight = (base.astype(mx.float32) + add[None]).astype(base.dtype)
    mx.eval(emb.weight)
    return base
