import sys, numpy as np, mlx.core as mx
from mlx_lm import load, generate
from score import MODEL
m, t = load(MODEL); S = np.load("neuron_scores_sel.npz")["score"]; flat = np.argsort(S.ravel())[::-1][:300]
qs = ["How does a knight move in chess?", "Who is Viswanathan Anand?", "Where did chess originate?", "What is checkmate?",
      "Who was King Arthur?", "How do you play ludo?", "What is the capital of France?"]
def ans(q):
    s = t.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
    return generate(m, t, s, max_tokens=50, verbose=False).split("<end_of_turn>")[0].strip().replace("\n", " ")[:200]
before = [ans(q) for q in qs]
for l in set(int(i // 6912) for i in flat):
    lin = m.model.layers[l].mlp.down_proj; mask = np.ones(6912, np.float32)
    mask[[int(i % 6912) for i in flat if i // 6912 == l]] = 0
    lin.weight = (lin.weight.astype(mx.float32) * mx.array(mask)[None]).astype(lin.weight.dtype)
for q, b in zip(qs, before):
    print("Q:", q, "\n  before:", b, "\n  after :", ans(q))
