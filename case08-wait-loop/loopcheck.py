"""Smoke test: does base Qwen3.5-0.8B loop in thinking mode on an easy GSM8K problem?
Sampling = MIT WSP eval protocol: temp 0.6, top_p 0.95, top_k 20, presence_penalty 1.5 over the
whole generation (vLLM semantics), thinking on. Capped at --max tokens for the smoke test."""
import argparse, re, time, json
import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

ap = argparse.ArgumentParser()
ap.add_argument("--max", type=int, default=4096)
ap.add_argument("--seeds", default="0,1,2")
a = ap.parse_args()

Q = ("Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins "
     "for her friends every day with four. She sells the remainder at the farmers' market daily for $2 "
     "per fresh duck egg. How much in dollars does she make every day at the farmers' market?\n"
     "Please reason step by step, and put your final answer within \\boxed{}.")
model, tok = load("mlx-community/Qwen3.5-0.8B-bf16")
prompt = tok.apply_chat_template([{"role": "user", "content": Q}], tokenize=False,
                                 add_generation_prompt=True, enable_thinking=True)
print(f"loaded, memory {mx.get_active_memory() / 1e9:.2f} GB")
sampler = make_sampler(temp=0.6, top_p=0.95, top_k=20)
for seed in map(int, a.seeds.split(",")):
    mx.random.seed(seed)
    procs = make_logits_processors(presence_penalty=1.5, presence_context_size=a.max + 4096)
    text, n, t0 = "", 0, time.time()
    for r in stream_generate(model, tok, prompt, max_tokens=a.max, sampler=sampler, logits_processors=procs):
        text += r.text; n = r.generation_tokens
    dt = time.time() - t0
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    print(json.dumps({"seed": seed, "tokens": n, "hit_cap": n >= a.max, "closed_think": "</think>" in text,
                      "answer": boxed[-1] if boxed else None, "wait_count": len(re.findall(r"\bWait\b", text)),
                      "tok_per_s": round(n / dt), "peak_GB": round(mx.get_peak_memory() / 1e9, 2)}))
    open(f"loopcheck_seed{seed}.txt", "w").write(text)
