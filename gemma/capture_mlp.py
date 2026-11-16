"""Record the input to one layer's mlp.down_proj (the MLP neurons) at the last token."""
import mlx.core as mx
import mlx.nn as nn


class TapLinear(nn.Module):
    def __init__(self, lin, store):
        super().__init__()
        self.lin, self.store = lin, store

    def __call__(self, x):
        self.store["a"] = x[0, -1].astype(mx.float32)
        return self.lin(x)


def mlp_acts(model, ids, layer):
    mlp = model.model.layers[layer].mlp
    store, orig = {}, mlp.down_proj
    mlp.down_proj = TapLinear(orig, store)
    try:
        model(mx.array([ids]))
    finally:
        mlp.down_proj = orig
    mx.eval(store["a"])
    return store["a"]


def mlp_acts_all(model, ids, layer):
    """Same, but every position: [len(ids), 6912]."""
    mlp = model.model.layers[layer].mlp
    store, orig = {}, mlp.down_proj

    class TapAll(TapLinear):
        def __call__(self, x):
            self.store["a"] = x[0].astype(mx.float32)
            return self.lin(x)

    mlp.down_proj = TapAll(orig, store)
    try:
        model(mx.array([ids]))
    finally:
        mlp.down_proj = orig
    mx.eval(store["a"])
    return store["a"]
