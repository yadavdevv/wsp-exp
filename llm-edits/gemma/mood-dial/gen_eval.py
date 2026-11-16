"""Generate answers at each dial strength (baked weights) for the test questions. Saves dial_answers.json."""
import json, numpy as np
from mlx_lm import load, generate
from bake import apply, LAYERS
from prompts import OPEN_TEST, TEST
m, tok = load("mlx-community/gemma-3-1b-it-bf16")
z = np.load("mood_edits.npz"); edits = {l: (z[f"u{l}"], z[f"r{l}"]) for l in LAYERS}
def ans(q, n):
    s = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
    return generate(m, tok, s, max_tokens=n, verbose=False).split("<end_of_turn>")[0].strip()
out, base = {}, None
for s in [-4, -3, -2, -1, 0, 1, 2, 3, 4]:
    base = apply(m, edits, float(s), base)
    out[s] = {"open": [ans(q, 70) for q in OPEN_TEST], "fact": [ans(q, 40) for q, _ in TEST]}
    ok = np.mean([any(k in a.lower() for k in kw) for a, (_, kw) in zip(out[s]["fact"], TEST)])
    print(f"s={s:+d}: factual correct {ok*100:.0f}% | {out[s]['open'][0][:110].replace(chr(10), ' ')}", flush=True)
json.dump(out, open("dial_answers.json", "w"), indent=1, ensure_ascii=False)
