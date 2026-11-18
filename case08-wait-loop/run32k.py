"""MIT-protocol run: 32,768-token cap, several problems decoded at once. Resumable.

    python run32k.py --arm bias|mit [--start 1 --n 100 --max 32768 --batch 8 --out results32k]

Differences from run_arm.py (the 8k pilot):
- presence penalty counts generated tokens only, as vLLM does. mlx-lm's built-in one also counts the
  prompt, so in the pilot the question's own numbers/words were penalised (all arms alike).
- continuous batching (BatchGenerator). Sampling is unchanged per token, but random draws are shared
  across the batch, so there is no per-problem seed: runs are not token-for-token reproducible.
"""
import argparse, json, os, re, time
import numpy as np
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import BatchGenerator
from mlx_lm.sample_utils import make_sampler

WAIT_IDS = [13784, 13428, 11158, 3655, 68620, 52565]  # Wait, " Wait", wait, " wait", WAIT, " WAIT"
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True, choices=["base", "bias", "mit"])
ap.add_argument("--start", type=int, default=1)
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--max", type=int, default=32768)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--bias", type=float, default=-5.0)
ap.add_argument("--out", default="results32k")
a = ap.parse_args()

model, tok = load("mlx-community/Qwen3.5-0.8B-bf16")
if a.arm == "mit":
    d = np.load("data/mit_edit_tensors.npz")
    model.load_weights([(f"language_model.model.layers.{i}.mlp.down_proj.weight",
                         mx.array(d[f"edit_model.language_model.layers.{i}.mlp.down_proj.weight"]).astype(mx.bfloat16))
                        for i in (15, 19)], strict=False)
    w = np.array(model.language_model.model.layers[15].mlp.down_proj.weight.astype(mx.float32))
    assert np.array_equal(w, d["edit_model.language_model.layers.15.mlp.down_proj.weight"]), "patch failed"


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


def output_presence_penalty(prompt_len, penalty=1.5):
    def proc(tokens, logits):
        gen = tokens[prompt_len:]
        if gen.size:
            logits[:, gen] -= penalty
        return logits
    return proc


def flush_lazy_counters(gen):
    # Batched caches update small counters (e.g. ArraysCache.left_padding -= 1) every step, and during
    # decoding nothing reads them, so the lazy graph grows ~1 buffer per layer per step until Metal's
    # buffer-count limit (499,000) is hit after ~35k steps. Evaluating them cuts the chain.
    arrs = [v for c in gen._generation_batch.prompt_cache for name in ("offset", "left_padding", "lengths")
            if isinstance(v := getattr(c, name, None), mx.array)]
    mx.eval(arrs)


bias_idx, bias_val = mx.array(WAIT_IDS), mx.array([a.bias] * len(WAIT_IDS))
problems = [json.loads(l) for l in open("data/gsm8k_test.jsonl")][a.start:a.start + a.n]
os.makedirs(f"{a.out}/text", exist_ok=True)
out_path = f"{a.out}/{a.arm}.jsonl"
done = {json.loads(l)["idx"] for l in open(out_path)} if os.path.exists(out_path) else set()

todo = []  # (idx, gold, prompt ids)
for k, p in enumerate(problems):
    idx = a.start + k
    if idx in done:
        continue
    gold = float(p["answer"].split("####")[-1].strip().replace(",", ""))
    prompt = tok.apply_chat_template([{"role": "user", "content": p["question"] + SUFFIX}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=True)
    todo.append((idx, gold, tok.encode(prompt)))

mx.random.seed(0)
mx.set_cache_limit(1 << 30)  # many distinct buffer sizes otherwise hit Metal's buffer-count limit
gen = BatchGenerator(model, max_tokens=a.max, stop_tokens=[[t] for t in tok.eos_token_ids],
                     sampler=make_sampler(temp=0.6, top_p=0.95, top_k=20),
                     completion_batch_size=a.batch, prefill_batch_size=min(a.batch, 4))
procs = []
for _, _, ids in todo:
    pl = [output_presence_penalty(len(ids))]
    if a.arm == "bias":
        pl.insert(0, lambda _, logits: logits.at[:, bias_idx].add(bias_val))
    procs.append(pl)
uids = gen.insert([ids for _, _, ids in todo], logits_processors=procs)
meta = dict(zip(uids, todo))
toks, t_start, t0, steps = {u: [] for u in uids}, {}, time.time(), 0

while responses := gen.next_generated():
    steps += 1
    if steps % 64 == 0:
        flush_lazy_counters(gen)
    if steps % 256 == 0:
        mx.clear_cache()
    for r in responses:
        t_start.setdefault(r.uid, time.time())
        if r.finish_reason is None or r.finish_reason == "length":
            toks[r.uid].append(r.token)
        if r.finish_reason is None:
            continue
        idx, gold, _ = meta[r.uid]
        text = tok.decode(toks[r.uid])
        n = len(toks[r.uid]) + (r.finish_reason == "stop")
        pred = to_number(last_boxed(text.split("</think>")[-1] if "</think>" in text else text))
        row = {"idx": idx, "gold": gold, "pred": pred, "correct": pred is not None and abs(pred - gold) < 1e-6,
               "tokens": n, "hit_cap": r.finish_reason == "length", "closed_think": "</think>" in text,
               "wait": len(re.findall(r"\bwait\b", text, re.I)),
               "substitutes": len(re.findall(r"\b(hmm|actually|re-read|double-check|let me check)\b", text, re.I)),
               "secs": round(time.time() - t_start[r.uid]), "peak_GB": round(mx.get_peak_memory() / 1e9, 2)}
        open(f"{a.out}/text/{a.arm}_{idx}.txt", "w").write(text)
        with open(out_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(a.arm, row, f"elapsed {round(time.time() - t0)}s", flush=True)
gen.close()
