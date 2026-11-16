"""Logit-lens reading of MIT's edit: which output tokens does it push down / up?

dW = edit - base is rank-1 per layer: dW ~ s u v^T (u in the 1024-d residual stream, v over the 3584 MLP
neurons). The edit adds s (v.a) u to the residual, where a = MLP activations. We read u through the final
RMSNorm gain and the unembedding, and take the sign of (v.a) from the base model's own looping output.

    python analysis/direction_vocab.py [results32k/text/base_2.txt]
"""
import sys
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

WAIT_IDS = [13784, 13428, 11158, 3655, 68620, 52565]
sample = sys.argv[1] if len(sys.argv) > 1 else "results32k/text/base_2.txt"

d = np.load("data/mit_edit_tensors.npz")
model, tok = load("mlx-community/Qwen3.5-0.8B-bf16")
lm = model.language_model
W_U = np.array((lm.lm_head.weight if hasattr(lm, "lm_head") else lm.model.embed_tokens.weight).astype(mx.float32))
g = np.array(lm.model.norm.weight.astype(mx.float32))
print(f"final-norm gain mean {g.mean():.3f}")


class Recorder(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner, self.seen = inner, []

    def __call__(self, x):
        self.seen.append(x)
        return self.inner(x)


layers = lm.model.layers
recs = {}
for L in (15, 19):
    recs[L] = Recorder(layers[L].mlp.down_proj)
    layers[L].mlp.down_proj = recs[L]
ids = tok.encode(open(sample).read())[:4096]
mx.eval(model(mx.array(ids)[None]))

dec = lambda xs: " ".join(repr(tok.decode([int(i)])) for i in xs)
for L in (15, 19):
    dW = d[f"edit_model.language_model.layers.{L}.mlp.down_proj.weight"] - d[f"base_model.language_model.layers.{L}.mlp.down_proj.weight"]
    U, S, Vt = np.linalg.svd(dW, full_matrices=False)
    u, v = U[:, 0], Vt[0]
    a = np.array(recs[L].seen[0][0].astype(mx.float32))  # (tokens, 3584)
    va = a @ v
    sign = np.sign(np.median(va))
    eff = sign * u  # direction the edit adds on typical tokens (times S0 |v.a|)
    score = W_U @ (g * eff)
    order = np.argsort(score)
    print(f"\nlayer {L}: s1={S[0]:.3f} s2={S[1]:.4f} neurons touched={(np.abs(v) > 1e-6).sum()} "
          f"| v.a over {len(va)} tokens: median {np.median(va):.3f}, share same sign {np.mean(np.sign(va) == sign):.0%}")
    print("  pushed DOWN:", dec(order[:30]))
    print("  pushed UP  :", dec(order[::-1][:30]))
    rank_down = {tok.decode([w]): int(np.where(order == w)[0][0]) for w in WAIT_IDS}
    print("  Wait-token rank among pushed-down (0 = most, of 248,320):", rank_down)
    wait_pos = np.isin(np.array(ids), WAIT_IDS)
    print(f"  mean v.a at Wait tokens {va[wait_pos].mean():.3f} vs other tokens {va[~wait_pos].mean():.3f}")
