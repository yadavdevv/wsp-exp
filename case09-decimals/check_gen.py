"""Edited vs unedited: scored margin AND greedy generation, same prompts."""
import numpy as np, mlx.core as mx
from mlx_lm import generate
from decimals import Scorer, make_pairs
from edit_yesno import build, apply, undo
sc = Scorer("mlx-community/gemma-3-1b-it-bf16"); u, _ = build(sc.model, sc.tok); r = np.load("r_v3.npy")
lin = sc.model.model.layers[25].mlp.down_proj
def ans(q):
    s = sc.tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
    return generate(sc.model, sc.tok, s, max_tokens=3, verbose=False).split("<end_of_turn>")[0].strip()
def marg(q):
    c = sc.ids(q); return sc.seq_logprob(c, "Yes") - sc.seq_logprob(c, "No")
qs = [f"Is {a} {w} than {b}? Answer Yes or No." for a, b, w in
      [("9.11", "9.9", "bigger"), ("9.9", "9.11", "bigger"), ("3.45", "3.5", "smaller"), ("5.8", "5.11", "bigger"), ("2.7", "2.65", "bigger"), ("6.4", "6.38", "smaller")]]
for tag in ["before", "after"]:
    if tag == "after": W0 = apply(sc.model, u, r, 1.0)
    print(tag, "weight checksum", float(lin.weight.astype(mx.float32).sum()))
    for q in qs: print(f"   {q:45s} margin(Yes-No) {marg(q):7.2f}   generated: {ans(q)}")
# generation-based accuracy on 80 held-out pairs (bigger phrasing), edited model
test = make_pairs(20, 2024); ok = 0
for p in test:
    g = ans(f"Is {p['a']} bigger than {p['b']}? Answer Yes or No.")
    ok += g.lower().startswith("yes" if p["bigger"] == p["a"] else "no")
print("edited, generated answers correct:", ok, "/", len(test))
undo(sc.model, W0); ok = 0
for p in test:
    g = ans(f"Is {p['a']} bigger than {p['b']}? Answer Yes or No.")
    ok += g.lower().startswith("yes" if p["bigger"] == p["a"] else "no")
print("unedited, generated answers correct:", ok, "/", len(test))
