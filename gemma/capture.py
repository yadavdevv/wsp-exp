"""Record the residual stream (output of every layer) at the last prompt token."""
import mlx.core as mx
import mlx.nn as nn


class Tap(nn.Module):
    def __init__(self, layer, store, i):
        super().__init__()
        self.layer, self.store, self.i = layer, store, i

    def __call__(self, x, mask=None, cache=None):
        h = self.layer(x, mask, cache)
        self.store[self.i] = h[0, -1].astype(mx.float32)
        return h


def residuals(model, ids):
    """Returns array [n_layers+1, d]: embedding (scaled) then each layer's output, at the last token."""
    store = {}
    inner = model.model
    orig = inner.layers
    inner.layers = [Tap(l, store, i + 1) for i, l in enumerate(orig)]
    try:
        x = mx.array([ids])
        h0 = inner.embed_tokens(x) * mx.array(inner.args.hidden_size ** 0.5, mx.bfloat16)
        store[0] = h0[0, -1].astype(mx.float32)
        model(x)
    finally:
        inner.layers = orig
    out = mx.stack([store[i] for i in range(len(orig) + 1)])
    mx.eval(out)
    return out
