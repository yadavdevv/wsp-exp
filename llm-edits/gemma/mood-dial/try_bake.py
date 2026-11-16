import os, numpy as np, mlx.core as mx
from mlx_lm import load, generate
from bake import apply, LAYERS
m, tok = load("mlx-community/gemma-3-1b-it-bf16")
z = np.load("mood_edits.npz"); edits = {l: (z[f"u{l}"], z[f"r{l}"]) for l in LAYERS}
for l in LAYERS:
    g = np.array(m.model.layers[l].post_feedforward_layernorm.weight.astype(mx.float32)) + 1
    print(f"layer {l+1}: min |gain| {np.abs(g).min():.4f}  |u| {np.linalg.norm(edits[l][0]):.1f}  finite {np.isfinite(edits[l][0]).all()}")
def ans(q):
    s = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
    return generate(m, tok, s, max_tokens=45, verbose=False).split("<end_of_turn>")[0].strip().replace("\n", " ")[:170]
base = None
for s in [float(x) for x in os.environ.get("S", "-4,0,4").split(",")]:
    base = apply(m, edits, s, base)
    print(f"BAKED s={s:+}: {ans(os.environ.get('Q', 'What do you think about Mondays?'))}")
