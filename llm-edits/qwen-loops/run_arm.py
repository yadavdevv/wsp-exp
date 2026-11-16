"""Evaluate one arm on GSM8K problems. Resumable: appends one JSON line per problem.

    python run_arm.py --arm base|bias|mit [--start 1 --n 30 --max 8192]

base: Qwen3.5-0.8B as released.  bias: same + logit bias on the six "Wait" tokens.
mit:  layers 15/19 down_proj replaced by RikiyaT/Qwen3.5-0.8B-weight-space-programming tensors.
Sampling = MIT eval protocol (temp 0.6, top_p 0.95, top_k 20, presence_penalty 1.5 over the whole
generation, thinking on). Seed = problem index, identical across arms (paired comparison).
"""
import argparse, json, os, re, time
import numpy as np
import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

WAIT_IDS = [13784, 13428, 11158, 3655, 68620, 52565]  # Wait, " Wait", wait, " wait", WAIT, " WAIT"
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True, choices=["base", "bias", "mit"])
ap.add_argument("--start", type=int, default=1)   # skip problem 0 (used in the loop check)
ap.add_argument("--n", type=int, default=30)
ap.add_argument("--max", type=int, default=8192)
ap.add_argument("--bias", type=float, default=-5.0)
a = ap.parse_args()

model, tok = load("mlx-community/Qwen3.5-0.8B-bf16")
if a.arm == "mit":
    d = np.load("data/mit_edit_tensors.npz")
    model.load_weights([(f"language_model.model.layers.{i}.mlp.down_proj.weight",
                         mx.array(d[f"edit_model.language_model.layers.{i}.mlp.down_proj.weight"]).astype(mx.bfloat16))
                        for i in (15, 19)], strict=False)
    w = np.array(model.language_model.model.layers[15].mlp.down_proj.weight.astype(mx.float32))
    assert np.array_equal(w, d["edit_model.language_model.layers.15.mlp.down_proj.weight"]), "patch failed"

problems = [json.loads(l) for l in open("data/gsm8k_test.jsonl")][a.start:a.start + a.n]
os.makedirs("results/text", exist_ok=True)
out_path = f"results/{a.arm}.jsonl"
done = {json.loads(l)["idx"] for l in open(out_path)} if os.path.exists(out_path) else set()
sampler = make_sampler(temp=0.6, top_p=0.95, top_k=20)


def last_boxed(s):
    i = s.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + 7, 1
    while j < len(s) and depth:
        depth += {"{": 1, "}": -1}.get(s[j], 0)
        j += 1
    return s[i + 7:j - 1] if depth == 0 else None


def to_number(s):
    if s is None:
        return None
    s = re.sub(r"\\text\{[^}]*\}|\\[a-z]+|[$,%\s]", "", s)
    m = re.search(r"-?\d+(\.\d+)?", s)
    return float(m.group()) if m else None


for k, p in enumerate(problems):
    idx = a.start + k
    if idx in done:
        continue
    gold = float(p["answer"].split("####")[-1].strip().replace(",", ""))
    prompt = tok.apply_chat_template([{"role": "user", "content": p["question"] + SUFFIX}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=True)
    mx.random.seed(idx)
    procs = make_logits_processors(logit_bias={i: a.bias for i in WAIT_IDS} if a.arm == "bias" else None,
                                   presence_penalty=1.5, presence_context_size=a.max + 4096)
    text, n, t0 = "", 0, time.time()
    for r in stream_generate(model, tok, prompt, max_tokens=a.max, sampler=sampler, logits_processors=procs):
        text += r.text
        n = r.generation_tokens
    pred = to_number(last_boxed(text.split("</think>")[-1] if "</think>" in text else text))
    row = {"idx": idx, "gold": gold, "pred": pred, "correct": pred is not None and abs(pred - gold) < 1e-6,
           "tokens": n, "hit_cap": n >= a.max, "closed_think": "</think>" in text,
           "wait": len(re.findall(r"\bwait\b", text, re.I)),
           "substitutes": len(re.findall(r"\b(hmm|actually|re-read|double-check|let me check)\b", text, re.I)),
           "secs": round(time.time() - t0), "peak_GB": round(mx.get_peak_memory() / 1e9, 2)}
    open(f"results/text/{a.arm}_{idx}.txt", "w").write(text)
    with open(out_path, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(a.arm, row, flush=True)
