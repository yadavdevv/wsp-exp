"""Reference only (not a weight edit): add s * mood vector to the residual stream after layer L, every position."""
import sys, os, numpy as np, mlx.core as mx, mlx.nn as nn
from mlx_lm import load, generate
m, tok = load("mlx-community/gemma-3-1b-it-bf16"); D = np.load("mood_dirs.npy")
class Add(nn.Module):
    def __init__(self, layer, v): super().__init__(); self.layer, self.v = layer, v
    def __call__(self, x, mask=None, cache=None): return self.layer(x, mask, cache) + self.v.astype(x.dtype)
def ans(q):
    s = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
    return generate(m, tok, s, max_tokens=45, verbose=False).split("<end_of_turn>")[0].strip().replace("\n", " ")[:170]
orig = list(m.model.layers)
for L in [int(x) for x in sys.argv[1:]]:
    for s in [float(x) for x in os.environ.get("S", "-1.5,-1,1,1.5").split(",")]:
        m.model.layers = orig[:L] + [Add(orig[L], mx.array(s * D[L + 1]))] + orig[L + 1:]
        print(f"after layer {L+1} s={s:+}: {ans(os.environ.get('Q', 'What do you think about Mondays?'))}")
m.model.layers = orig
