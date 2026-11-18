"""Bake 'add s * mood vector after layer 10' into weights: rank-1 edits of mlp.down_proj in layers 4..10.
  down_proj_l' = down_proj_l + (s/K) u_l r_l^T
  r_l : read vector with r_l . a ~ 1 for every token (ridge least squares on MLP activations of ordinary text + chat)
  u_l : mood vector pulled back through that layer's post-MLP RMSNorm:  u_l = v * rms_l / gain_l
        (rms_l = typical RMS of the down_proj output, so after the norm the layer adds ~ v/K)"""
import json, sys, numpy as np, mlx.core as mx, mlx.nn as nn
from mlx_lm import load
LAYERS = list(range(3, 10))                  # layer indices 3..9 = layers 4..10
K = len(LAYERS)


class Rec(nn.Module):
    def __init__(self, lin, store, i):
        super().__init__(); self.lin, self.store, self.i = lin, store, i
    def __call__(self, x):
        y = self.lin(x); self.store[self.i] = (x[0].astype(mx.float32), y[0].astype(mx.float32)); return y


def collect(m, tok, texts):
    A = {l: [] for l in LAYERS}; Y = {l: [] for l in LAYERS}
    for ids in texts:
        store = {}; orig = {l: m.model.layers[l].mlp.down_proj for l in LAYERS}
        for l in LAYERS: m.model.layers[l].mlp.down_proj = Rec(orig[l], store, l)
        try: m(mx.array([ids]))
        finally:
            for l in LAYERS: m.model.layers[l].mlp.down_proj = orig[l]
        for l in LAYERS:
            a, y = store[l]; mx.eval(a, y); A[l].append(np.array(a)[1:]); Y[l].append(np.array(y)[1:])
    return {l: np.concatenate(A[l]) for l in LAYERS}, {l: np.concatenate(Y[l]) for l in LAYERS}


def fit(m, tok, v):
    chat = lambda u: tok.encode(tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True), add_special_tokens=False)
    qs = [json.loads(l)["question"] for l in open("../case08-wait-loop/data/gsm8k_test.jsonl")][400:460]
    ans = json.load(open("train_answers.json"))
    from prompts import TRAIN
    texts = [chat(q) if i % 2 else tok.encode(q) for i, q in enumerate(qs)]
    texts += [chat(q) + tok.encode(a, add_special_tokens=False) for q, a in zip(TRAIN, ans["cheer"][:20] + ans["grump"][20:])]
    A, Y = collect(m, tok, texts)
    edits = {}
    for l in LAYERS:
        X = A[l]; mu = 1e-2 * np.trace(X.T @ X) / X.shape[1]
        r = np.linalg.solve(X.T @ X + mu * np.eye(X.shape[1]), X.T @ np.ones(len(X)))
        rd = X @ r
        g = np.array(m.model.layers[l].post_feedforward_layernorm.weight.astype(mx.float32)) + 1
        rms = float(np.median(np.sqrt((Y[l] ** 2).mean(1))))
        u = v * rms * g / (g ** 2 + 0.01)            # = v * rms / g where the gain is not tiny; ~0 where the norm blocks writing
        edits[l] = (u.astype(np.float32), r.astype(np.float32))
        print(f"layer {l+1}: read value on fit text mean {rd.mean():.3f} sd {rd.std():.3f} | typical rms {rms:.2f}")
    return edits


def apply(m, edits, s, base=None):
    base = base or {l: m.model.layers[l].mlp.down_proj.weight for l in edits}
    for l, (u, r) in edits.items():
        W = base[l]
        m.model.layers[l].mlp.down_proj.weight = (W.astype(mx.float32) + mx.array((s / K) * np.outer(u, r))).astype(W.dtype)
    mx.eval(m.parameters())
    return base


if __name__ == "__main__":
    m, tok = load("mlx-community/gemma-3-1b-it-bf16"); D = np.load("mood_dirs.npy")
    edits = fit(m, tok, D[10])
    np.savez("mood_edits.npz", **{f"u{l}": e[0] for l, e in edits.items()}, **{f"r{l}": e[1] for l, e in edits.items()})
