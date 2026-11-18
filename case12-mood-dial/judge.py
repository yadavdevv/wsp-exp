"""Tone judge: a different model (Qwen3.5-0.8B, thinking off). Score = expected value of the digit 1-5
(1 very grumpy ... 5 very cheerful) from its next-token probabilities. Also runs on the two instructed sets as a sanity check."""
import json, numpy as np, mlx.core as mx
from mlx_lm import load
from prompts import OPEN_TEST
jm, jt = load("mlx-community/Qwen3.5-0.8B-bf16")
DIG = [jt.encode(str(i), add_special_tokens=False)[0] for i in range(1, 6)]
def tone(q, a):
    msg = (f"A user asked: \"{q}\"\nAn assistant replied: \"{a[:600]}\"\n\nHow cheerful or grumpy is the tone of the reply? "
           "Answer with a single digit: 1 = very grumpy/annoyed, 3 = neutral, 5 = very cheerful/happy.")
    try: s = jt.apply_chat_template([{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError: s = jt.apply_chat_template([{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True)
    lp = jm(mx.array([jt.encode(s, add_special_tokens=False)]))[0, -1].astype(mx.float32)
    p = np.array(mx.softmax(lp[mx.array(DIG)]))
    return float(p @ np.arange(1, 6))
ans = json.load(open("dial_answers.json")); tr = json.load(open("train_answers.json"))
from prompts import TRAIN
print("sanity, instructed train answers: cheerful", round(np.mean([tone(q, a) for q, a in zip(TRAIN[:10], tr["cheer"][:10])]), 2),
      "grumpy", round(np.mean([tone(q, a) for q, a in zip(TRAIN[:10], tr["grump"][:10])]), 2))
res = {}
for s, v in ans.items():
    res[s] = float(np.mean([tone(q, a) for q, a in zip(OPEN_TEST, v["open"])]))
    print(f"s={int(s):+d}: tone {res[s]:.2f}")
json.dump(res, open("dial_tone.json", "w"), indent=1)
