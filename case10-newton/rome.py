"""ROME-style rank-1 fact edit for Gemma 3 1B (MLX).

At layer L, mlp.down_proj maps 'keys' k (6912 MLP neurons) to outputs. Edit:
    W' = W + delta (C^-1 k*)^T / (k*^T C^-1 k*)
  k*    = mean key at the subject's last token (" Newton") over a few contexts
  C     = uncentred covariance of keys over ordinary text (so other keys are disturbed as little as possible)
  delta = vector added to down_proj's output at the subject token, optimised (gradient on delta only,
          model weights frozen) so the model says the new fact.
"""
import json, numpy as np, mlx.core as mx, mlx.nn as nn
from mlx_lm import load

MODEL = "mlx-community/gemma-3-1b-it-bf16"


def plain_sdpa(queries, keys, values, cache, scale, mask, sinks=None):
    """Differentiable attention (the fused kernel has no gradient for these shapes)."""
    H, Hk = queries.shape[1], keys.shape[1]
    if H != Hk:
        keys = mx.repeat(keys, H // Hk, axis=1); values = mx.repeat(values, H // Hk, axis=1)
    s = (queries * scale) @ keys.swapaxes(-1, -2)
    T, S = s.shape[-2], s.shape[-1]
    if isinstance(mask, str):                     # "causal"
        m = mx.arange(S)[None, :] <= (mx.arange(T)[:, None] + (S - T))
        s = mx.where(m, s, mx.array(-1e9, s.dtype))
    elif mask is not None:
        s = mx.where(mask, s, mx.array(-1e9, s.dtype)) if mask.dtype == mx.bool_ else s + mask
    return mx.softmax(s.astype(mx.float32), axis=-1).astype(values.dtype) @ values


class Inject(nn.Module):
    """Wraps down_proj: records its input at `pos` and adds `delta` to its output there."""
    def __init__(self, lin):
        super().__init__()
        self.lin, self.pos, self.delta, self.rec, self.rec_all = lin, None, None, None, False

    def __call__(self, x):
        y = self.lin(x)
        if self.rec_all:
            self.rec = x[0]
        if self.pos is not None:
            self.rec = x[0, self.pos]
            if self.delta is not None:
                onehot = (mx.arange(y.shape[1]) == self.pos)[None, :, None]
                y = y + onehot * self.delta.astype(y.dtype)
        return y


class Editor:
    def __init__(self, layer):
        self.model, self.tok = load(MODEL)
        self.model.freeze()
        self.L = layer
        self.mlp = self.model.model.layers[layer].mlp
        self.orig = self.mlp.down_proj
        self.inj = Inject(self.orig)
        self.mlp.down_proj = self.inj

    def chat(self, user):
        s = self.tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
        return self.tok.encode(s, add_special_tokens=False)

    @staticmethod
    def last(ids, tid):
        return max(i for i, x in enumerate(ids) if x == tid)

    def key(self, ids, pos):
        self.inj.pos, self.inj.delta = pos, None
        self.model(mx.array([ids]))
        k = self.inj.rec.astype(mx.float32); mx.eval(k)
        self.inj.pos = None
        return np.array(k)

    def keys_all(self, ids):
        """keys at every position (for the covariance)."""
        self.inj.rec_all = True
        try:
            self.model(mx.array([ids]))
        finally:
            self.inj.rec_all = False
        k = self.inj.rec.astype(mx.float32); mx.eval(k); return np.array(k)

    def optimise_delta(self, items, steps=60, lr=0.5, l2=0.02, log=print):
        """items: list of (ids, subject_pos, target_ids). Minimise NLL of targets given prompt with delta injected."""
        d = self.model.model.args.hidden_size
        delta = mx.zeros((d,))
        def loss_fn(delta):
            tot = 0.0
            for ids, pos, tgt in items:
                x = mx.array([ids + tgt[:-1]])
                self.inj.pos, self.inj.delta = pos, delta
                lp = self.model(x)[0].astype(mx.float32)
                lp = lp - mx.logsumexp(lp, -1, keepdims=True)
                n = len(ids)
                tot = tot - sum(lp[n - 1 + j, tgt[j]] for j in range(len(tgt))) / len(tgt)
            return tot / len(items) + l2 * mx.sum(delta * delta)
        import mlx_lm.models.gemma3_text as g3
        fast = g3.scaled_dot_product_attention
        g3.scaled_dot_product_attention = plain_sdpa
        vg = mx.value_and_grad(loss_fn)
        m = mx.zeros((d,)); v = mx.zeros((d,))
        best = (float("inf"), delta)
        for s in range(steps):                        # Adam on delta only; keep the best
            loss, g = vg(delta)
            if loss.item() < best[0]:
                best = (loss.item(), delta)
            m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
            delta = delta - lr * (m / (1 - 0.9 ** (s + 1))) / (mx.sqrt(v / (1 - 0.999 ** (s + 1))) + 1e-8)
            mx.eval(delta, loss)
            if s % 25 == 0 or s == steps - 1:
                log(f"   step {s:3d} loss {loss.item():.3f} |delta| {mx.linalg.norm(delta).item():.2f}")
        g3.scaled_dot_product_attention = fast
        self.inj.pos, self.inj.delta = None, None
        log(f"   best loss {best[0]:.3f}")
        return np.array(best[1])

    def apply(self, dW):
        W = self.orig.weight
        self.orig.weight = (W.astype(mx.float32) + mx.array(dW)).astype(W.dtype)
        mx.eval(self.orig.weight)
        return W

    def undo(self, W):
        self.orig.weight = W

    def generate(self, user, max_tokens=60):
        from mlx_lm import generate
        s = self.tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
        return generate(self.model, self.tok, s, max_tokens=max_tokens, verbose=False).split("<end_of_turn>")[0].strip()


def rank1(delta, kstar, Cinv_k):
    return np.outer(delta, Cinv_k) / float(kstar @ Cinv_k)
