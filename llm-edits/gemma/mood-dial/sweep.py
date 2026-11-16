import sys, numpy as np
from mlx_lm import load, generate
from dial import set_mood
m, tok = load("mlx-community/gemma-3-1b-it-bf16"); D = np.load("mood_dirs.npy"); E0 = m.model.embed_tokens.weight
def ans(q):
    s = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
    return generate(m, tok, s, max_tokens=40, verbose=False).split("<end_of_turn>")[0].strip().replace("\n", " ")[:150]
for L in [int(x) for x in sys.argv[1:]]:
    for s in [float(x) for x in __import__("os").environ.get("S", "-4,-2,2,4").split(",")]:
        set_mood(m, D[L], s, E0)
        print(f"L{L} s={s:+}: {ans('What is the capital of Japan?')}")
m.model.embed_tokens.weight = E0
