"""Longer answers and two follow-up conversations at each dial setting, for the article's dial. Saves dial_showcase.json.
Sampled (temp 0.7, top-p 0.95, repetition penalty 1.15, seed 0 per answer), 220 tokens per answer."""
import json, numpy as np, mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from bake import apply, LAYERS
SINGLE = ["My train is late. What should I do?", "Can you help me plan a birthday party?", "What do you think of homework?"]
CHATS = [["Who would win in a fight: the blue Hulk or the green Avatar guy?",
          "Wait, isn't the Hulk green and the Avatar people blue?",
          "Okay, so who actually wins?"],
         ["I just got a puppy!", "What should I name him?", "He chewed my favourite shoes. Now what?"]]
m, tok = load("mlx-community/gemma-3-1b-it-bf16")
z = np.load("mood_edits.npz"); edits = {l: (z[f"u{l}"], z[f"r{l}"]) for l in LAYERS}
def ans(msgs):
    mx.random.seed(0)
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return generate(m, tok, s, max_tokens=220, verbose=False, sampler=make_sampler(temp=0.7, top_p=0.95),
                    logits_processors=make_logits_processors(repetition_penalty=1.15, repetition_context_size=64)
                    ).split("<end_of_turn>")[0].strip()
out, base = {"single": SINGLE, "chats": CHATS, "a": {}}, None
for s in [-4, -3, -2, -1, 0, 1, 2, 3, 4]:
    base = apply(m, edits, float(s), base)
    single = [ans([{"role": "user", "content": q}]) for q in SINGLE]
    chats = []
    for turns in CHATS:
        msgs, said = [], []
        for q in turns:
            msgs.append({"role": "user", "content": q}); a = ans(msgs)
            msgs.append({"role": "assistant", "content": a}); said.append(a)
        chats.append(said)
    out["a"][s] = {"single": single, "chats": chats}
    print(f"s={s:+d} | {chats[0][0][:120]!r}", flush=True)
json.dump(out, open("dial_showcase.json", "w"), indent=1, ensure_ascii=False)
