# Case 8 · A model that can't stop saying "Wait": is it just a logit bias?

> **Read section 9 first.** The 8k pilot directly below is superseded: its cap and its presence-penalty setting
> were both wrong (see section 9, finding 1). It is kept only as a record.

## 8k pilot (2026-11-14, superseded)

Qwen3.5-0.8B (MLX bf16; the edited layers 15 and 19 are bit-identical to Qwen's release), GSM8K test problems 2-31 (problem 1 was used for
the loop check), one sample each, thinking on, MIT eval sampling (temp 0.6, top_p 0.95, top_k 20,
presence_penalty 1.5 over the whole generation), **8,192-token cap**, same seed per problem in every arm.

| Arm | Correct | 95% CI | Hit cap | Finished but wrong | Verbatim-loop endings | "Wait" per answer |
|---|---|---|---|---|---|---|
| base | 3/30 (10%) | 3-26% | 27 | 0 | 18 | 77.7 |
| **bias**: -5 on 6 "Wait" tokens | **13/30 (43%)** | 27-61% | 17 | 0 | 9 | 0.1 |
| **mit**: released 131k-weight edit | 9/30 (30%) | 17-48% | 20 | 1 | **1** | 28.9 |

- vs base: bias fixed 10, broke 0. mit fixed 8, broke 2.
- bias vs mit head-to-head: both right 6, only bias 7, only mit 3 (sign test p = 0.34, not significant).
- Of the answers that finished in time, 26 of 26 were correct for base and bias, and 9 of 10 for mit.
  At this cap, accuracy is almost entirely whether the model stops in time, not whether it reasons better.
- The edit does what MIT says it does: it nearly removes the literal repetition loops (18 -> 1 endings). But
  the answers then ramble without repeating and still run out of tokens. The bias halves loops (18 -> 9) and
  gets more answers finished. With "Wait" banned, "However" rises from 73 to 138, but the text stays clean
  (no Chinese characters, no garbling).
- The bias strength (-5) and the token list were fixed before the run and never tuned.

## What this does and doesn't show
- Supports the Reddit critique: on this test a one-line logit bias matched or beat the 131k-weight edit.
- Doesn't refute the edit: with only 30 problems the difference isn't significant, and MIT measured at 32k
  tokens, where the edit's shorter reasoning (mean 4,688 tokens when finished) may pay off more. Also one
  sample per problem, GSM8K only, one bias strength.
- Neither intervention makes the model smarter: both make it stop. When base finishes, it is already right.

Loop detector: regex for a 40+ character block repeated 4+ times in the last 4,000 characters.

---

# 9. Main result — MIT protocol, 32k cap (2026-11-15; supersedes the 8k pilot above)

`run32k.py`: GSM8K test problems 1–100 (file order; #0 used for the smoke test), 1 sample each, MIT's exact
prompt, thinking on, temp 0.6 / top_p 0.95 / top_k 20 / presence 1.5 (output tokens only, like vLLM),
32,768 new tokens, batch 8 on MLX. Rows from crashed attempts are archived in `results32k_crashed/` and unused.

| Arm | Correct | 95% CI | Strict* | Hit cap | Finished but wrong | Mean tokens | "Wait" / answer |
|---|---|---|---|---|---|---|---|
| base | 15 | 9–23% | 14 | 86 | 0 | 29,054 | 548.1 |
| bias (−5 on 6 Wait tokens) | 54 | 44–63% | 51 | 42 | 7 | 17,207 | 0.2 |
| **MIT edit** | **65** | 55–74% | 65 | 2 | 33 | 12,601 | 60.4 |

*Strict = only answers that closed `</think>` (the default grader also reads the last \boxed{} inside an unfinished think).

Paired (sign test): bias vs base 40–1 (p = 4e-11); MIT vs base 52–2 (p = 2e-13); **MIT vs bias 15–4 (p = 0.019)**.

Reference (Takehi et al., "Sparks of Weight-Space Programming", MIT CSAIL blog, rikiyat.github.io, 2 Nov 2026): GSM8K
base 13.8%, edited 67.1% (2 samples per problem, 32,768-token cap). Their 93% "hit the token limit" figure is for
MATH-500, not GSM8K. Ours (GSM8K 1-100, 1 sample): 15% / 65%, with 86% of base answers hitting the cap — the
accuracy replicates; the cap share is not directly comparable.

## What the edit is, exactly
Each Qwen layer's MLP computes `down_proj(silu(gate_proj(x)) * up_proj(x))`; `down_proj` is 1024 x 3584 and
each of its 3,584 columns is the vector one MLP neuron writes into the residual stream. MIT's edit:
| | layer 15 | layer 19 |
|---|---|---|
| columns (neurons) changed | 64 of 3,584 | 64 of 3,584 |
| entries changed | 65,260 | 64,928 |
| rank of the change | 1 (100% of its energy) | 1 |
| change / whole matrix (Frobenius) | 0.106 | 0.081 |
| change / those 64 columns | 0.85 | 0.62 |
| largest single change vs mean \|weight\| | 0.146 vs 0.0066 | 0.215 vs 0.0071 |
i.e. `W' = W - s u v^T` with v non-zero on 64 neurons: those neurons' outputs lose (or gain) their
component along one residual direction u. Small for the matrix, heavy for those 64 neurons.

## Findings
1. **The 8k pilot reversed.** At 8k the bias looked better (43 vs 30); under the real protocol the MIT edit
   wins clearly. Two pilot flaws: the cap truncated the edit's finishing answers, and mlx-lm's presence penalty
   also counted prompt tokens.
2. **MIT's GSM8K claim replicates** (65% vs 67.1%).
3. **The edit does not work by removing "Wait".** Its answers still contain ~60 per answer; 98/100 finish.
   Logit-lens of the edit direction (`analysis/direction_vocab.py`): it pushes UP "proceed" words
   (first, thus, result, value, accordingly, 即可) and DOWN junk fragments; Wait tokens sit mid-vocabulary.
   It is active on 80–90% of tokens, not specifically on Wait. Caveat: logit lens at layers 15/19 of 24 is approximate.
4. **Banning "Wait" ≠ the edit.** The doubting continues in other words; 42/100 still hit the cap.
5. **The edit makes the model decisive, not smarter:** 33 finished-but-wrong vs 7 for bias; accuracy among
   finished answers: base 100% (14/14), bias 88% (51/58), MIT 66% (65/98).
6. The MIT post reports no simple baseline (logit bias, token ban, steering vector, penalty tuning, or random edit of
   the same size); it says "The baselines here are not there to rank methods against each other on equal terms."
   (checked 2026-11-17, incl. its 12 Nov "Response to Comments"). The logit-bias comparison here is ours.

## Still to do
- ~~Regrade with math-verify 0.9.0~~ done (`analysis/regrade_mathverify.py`): 15 / 54 / 65, **0 disagreements** with our grader on all 300 answers.
- ~~Layer-19 patch check~~ done in the audit (2026-11-17): after patching, layers 15 and 19 equal MIT's released
  tensors exactly, and before patching they equal Qwen's release exactly.
- **Not done:** random equal-norm edit on the same 64 columns (the control used by Hu et al., arXiv 2611.09382)
  at 32k. Do not claim the edit beats a random edit of the same size.
- Caveats: 100 problems, 1 sample, MLX vs vLLM numerics, the bias strength was not tuned.
