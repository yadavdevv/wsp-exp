"""Generate cheerful and grumpy answers to TRAIN questions, then read the residual stream over the answer tokens
WITHOUT the instruction in the prompt. mood vector (per layer) = mean(cheerful) - mean(grumpy)."""
import sys, json, numpy as np, mlx.core as mx
sys.path.insert(0, "..")
from mlx_lm import load, generate
from prompts import TRAIN, CHEER, GRUMP
m, tok = load("mlx-community/gemma-3-1b-it-bf16"); inner = m.model
def chat_str(u): return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
ans = {"cheer": [], "grump": []}
for q in TRAIN:
    for k, ins in [("cheer", CHEER), ("grump", GRUMP)]:
        ans[k].append(generate(m, tok, chat_str(ins + q), max_tokens=80, verbose=False).split("<end_of_turn>")[0].strip())
json.dump(ans, open("train_answers.json", "w"), indent=1, ensure_ascii=False)
from capture import Tap
def resid(ids, start):
    store = {}
    class TapAll(Tap):
        def __call__(self, x, mask=None, cache=None):
            h = self.layer(x, mask, cache); self.store[self.i] = h[0, start:].astype(mx.float32); return h
    orig = inner.layers; inner.layers = [TapAll(l, store, i + 1) for i, l in enumerate(orig)]
    try:
        x = mx.array([ids]); m(x)
        e = (inner.embed_tokens(x) * mx.array(1152 ** 0.5, mx.bfloat16))[0, start:].astype(mx.float32)
    finally: inner.layers = orig
    R = mx.stack([e] + [store[i] for i in range(1, len(orig) + 1)]).mean(1); mx.eval(R); return np.array(R)
S = {}
for k in ans:
    rs = []
    for q, a in zip(TRAIN, ans[k]):
        p = tok.encode(chat_str(q), add_special_tokens=False); ids = p + tok.encode(a, add_special_tokens=False)
        rs.append(resid(ids, len(p)))
    S[k] = np.stack(rs)                     # [40, 27, 1152]  (layer 0 = embeddings)
np.savez("mood_resid.npz", **S)
D = S["cheer"].mean(0) - S["grump"].mean(0)
np.save("mood_dirs.npy", D.astype(np.float32))
print("|mood diff| per layer:", [round(float(np.linalg.norm(d)), 1) for d in D])
print("sample cheer:", ans["cheer"][0][:150].replace("\n", " ")); print("sample grump:", ans["grump"][0][:150].replace("\n", " "))
