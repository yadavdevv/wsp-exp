"""Erase a direction from everything that writes into the residual stream.
Each layer adds  post_norm(o_proj(...))  and  post_norm(down_proj(...)).  RMSNorm multiplies by a gain g, so the
written vector is perpendicular to d exactly when the matrix output is perpendicular to e = g*d:
    W <- W - e (e^T W),  e = (g*d)/|g*d|."""
import numpy as np, mlx.core as mx


def _proj(lin, gain, d):
    e = (np.array(gain.astype(mx.float32)) + 1.0) * d; e /= np.linalg.norm(e)
    W = np.array(lin.weight.astype(mx.float32))               # [out=1152, in]
    lin.weight = mx.array(W - np.outer(e, e @ W)).astype(lin.weight.dtype)


def erase(model, dirs, layers=None, embed=False):
    """dirs: [n_layers, d] (per-layer) or [d] (one direction everywhere). Returns a backup for restore()."""
    inner = model.model; backup = {}
    for i, L in enumerate(inner.layers):
        if layers is not None and i not in layers: continue
        d = dirs if dirs.ndim == 1 else dirs[i]
        backup[(i, "o")] = L.self_attn.o_proj.weight; backup[(i, "down")] = L.mlp.down_proj.weight
        _proj(L.self_attn.o_proj, L.post_attention_layernorm.weight, d)
        _proj(L.mlp.down_proj, L.post_feedforward_layernorm.weight, d)
    if embed:
        d = dirs if dirs.ndim == 1 else dirs[0]
        E = np.array(inner.embed_tokens.weight.astype(mx.float32)); backup["E"] = inner.embed_tokens.weight
        inner.embed_tokens.weight = mx.array(E - np.outer(E @ d, d)).astype(inner.embed_tokens.weight.dtype)
    mx.eval(model.parameters())
    return backup


def restore(model, backup):
    inner = model.model
    for k, W in backup.items():
        if k == "E": inner.embed_tokens.weight = W
        elif k[1] == "o": inner.layers[k[0]].self_attn.o_proj.weight = W
        else: inner.layers[k[0]].mlp.down_proj.weight = W
