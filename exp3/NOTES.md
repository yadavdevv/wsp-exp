# Exp 3 notes — verified facts (2026-11-14)

## Sources checked
- Qwen/Qwen3.5-0.8B model card (Apache-2.0) says verbatim: "In thinking mode, we have observed that when
  using the recommended sampling parameters, Qwen3.5-0.8B is more prone to entering thinking loops compared
  to other Qwen3.5 models, which may prevent it from terminating generation properly."
- **Two different edits exist — don't conflate them:**
  - MIT CSAIL blog (Takehi et al.): layers 15 and 19 MLP `down_proj`, one direction subtracted from 64
    output columns each = 131,072 weights. Released: `RikiyaT/Qwen3.5-0.8B-weight-space-programming`.
    Eval protocol: vLLM, thinking on, temp 0.6, top_p 0.95, top_k 20, **presence_penalty 1.5**, 32,768 max
    tokens. Base -> edited: MATH-500 7.9 -> 48.1, GSM8K 13.8 -> 67.1, token-limit hits 93% -> 33%.
  - Shanghai AI Lab paper (arXiv 2611.09382): layer 23 `down_proj`, column 507, 1,023 values, MATH-500
    8.35 -> 37.80. **Not released** — github.com/WSP-Scope/WSP-Scope-Project contains only a PDF.
- The baseline in the MIT eval already used presence_penalty 1.5 (a standard anti-repetition knob)
  and still hit the token limit 93% of the time.

## Local setup
- M4 Pro, 24 GB. mlx-lm 0.32.2, `mlx-community/Qwen3.5-0.8B-bf16`. Model: 24 layers (linear attention,
  full attention every 4th), hidden 1024, MLP 3584, vocab 248,320.
- Peak memory 1.68 GB, ~120 tokens/s.
- mlx-lm's presence penalty defaults to the last 20 tokens; set `presence_context_size` to cover the
  whole generation to match vLLM.

## Loop check (`loopcheck.py`, GSM8K #1 "Janet's ducks", answer 18, cap 4,096, MIT sampling)
All 3 seeds hit the cap without closing `</think>`. Three different failure modes:
- seed 0: settled on the **wrong** answer (26, forgot the muffin eggs) and kept second-guessing it.
- seed 1: found the **right** answer (18, mentioned 8 times) but could not stop re-checking.
- seed 2: literal verbatim loop ("Wait, maybe the 'remainder' refers to the muffins?" x many).
=> "Stop doubting" can only rescue seed 1-type failures; seed 0 would become confidently wrong.

## 32k run (`run32k.py`, MIT protocol) — methodology findings
- **Presence-penalty window.** mlx-lm's built-in presence penalty (both `stream_generate` and
  `BatchGenerator`) counts the *prompt* tokens too (verified: the processor sees 13 tokens before the first
  generated one for a 13-token prompt). vLLM counts output tokens only. So the 8k pilot also penalised the
  question's own numbers/words (all arms alike). `run32k.py` uses an output-only penalty.
- **Batched-cache leak (mlx-lm 0.32.2).** With continuous batching, `ArraysCache.advance` does
  `left_padding -= N` every step for the 18 linear-attention layers; nothing evaluates it during decoding,
  so the lazy graph grows until Metal's buffer-count limit (499,000) -> crash after ~35-40k steps
  ("[metal::malloc] Resource limit (499000) exceeded"). Related: ml-explore/mlx issue #3866. Not a memory
  problem (active memory stayed ~2 GB). Fix: `mx.eval` the cache counters every 64 steps.
- **Crash-resume is not neutral.** A crash kills the still-running (= longest) generations; resampling them
  favours shorter, likelier-correct samples. Rows from crashed runs are archived in `results32k_crashed/`
  and NOT used. The clean run never retries.
- Batching: random draws are shared across the batch, so there is no per-problem seed (not token-for-token
  reproducible). Throughput ~585 tok/s at batch 8 vs ~128 single-stream (short context).
