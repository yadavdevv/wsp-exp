# Audit before publishing (2026-11-17)

Folder names below are the current ones (the experiments were run as exp1/exp2/exp3/life/flight/robots/drive/gemma
and renamed afterwards; see the git history).

Every number cited in the `RESULTS.md` files was re-checked against the saved outputs or by re-running the
evaluation. What was checked, how, and what was corrected:

| Experiment | How it was checked | Result |
|---|---|---|
| bug-hunt | `kit/evaluate.py --seed 999` (the grader's seed) on the original and fixed nets; parameter diffs | all probe values and parameter counts reproduce |
| tictactoe | `ttt.full_report` on the hand-written net and baselines | 0 losing lines, optimal-move rate 1.0000, 4268/732/0, 904 nonzero weights: reproduce |
| qwen-loops | table recomputed from `results32k/*.jsonl`; math-verify regrade; edit tensors re-analysed; patched weights compared with MIT's release | all numbers reproduce, 0 regrade disagreements; layers 15 and 19 bit-identical to MIT's edit after patching and to Qwen's release before |
| game-of-life | `ca.py` / `life.py` regrade of all famous nets and secret net 3 against the literal spec; SHA-1 of committed nets; training logs tallied | reproduce (net3 94.8% cells, 23.8% boards; 2/20 minimal, 15 collapsed) |
| lander | `compare.py` re-run on the same 2,000 starts; v1 patches; gust sweep | reproduce exactly |
| walker | 8 table cells re-run (30 episodes each); patch hashes | reproduce |
| driver | `edits_results.json` vs text; weight diffs per edit; mirror edit tested on 2,000 random inputs | reproduce; mirror deviation 2.4e-15 (driving simulations themselves not re-run) |
| gemma/decimals | re-scored with every training pair removed (`rescore_clean.py`) | changes under 1 point |
| gemma/forget-chess, mood-dial | chess quiz re-run (unedited, 300 neurons off); mood table recomputed from saved answers | reproduce |
| all edits | weight changes counted from the actual weights (`llm-edits/gemma/edits_all.py`) | see table below |

## Corrections made
1. **Citations.** arXiv 2611.09382 is Hu et al. (Shanghai AI Lab / Tongji / Fudan), not the MIT work. The MIT CSAIL
   work is a blog post: Takehi, Teoh, Zhang, Khattab, "Sparks of Weight-Space Programming", rikiyat.github.io,
   2 Nov 2026.
2. **qwen-loops:** MIT's "93% of base answers hit the token limit" is for MATH-500, not GSM8K; the post does not state
   "full GSM8K" or the inference engine. "MIT never compared against a logit bias" is narrowed to: the MIT post
   reports no simple baseline (checked, including its 12 Nov response section). The layer-19 patch check is
   done. The random equal-norm control was **not** run.
3. **driver:** the unedited left-hand baseline is 1/40 routes (2.5%). The "school edit, no school zones" row is
   identical to the unedited car by construction (the edit only reads the school input, which is 0 there), so it
   is not evidence about side effects. The school-speeding threshold is 20 km/h + 1 m/s.
4. **lander:** the stuck-thruster patch lands 0% in calm air (was "flies badly").
5. **gemma/decimals:** 10 of 200 test pairs overlapped training; the clean numbers are now the ones to cite.
6. **gemma/newton:** "direct" questions are paraphrases of the trained facts; "which national team" is close to a
   trained fact, so the strict reasoning score is 1/4.
7. **bug-hunt:** numbers are at the grader's seed 999 (other seeds differ by up to ~0.01).

## Weight changes (measured)
| Edit | Weights changed | Share of the model | Numbers defining the edit |
|---|---|---|---|
| Gemma mood dial (s = 2, 7 matrices) | 41,777,331 | 3.21% | 56,448 |
| Gemma decimal fix (1 matrix) | 7,456,750 | 0.57% | 8,064 |
| Gemma Newton rewrite (1 matrix) | 5,452,090 | 0.42% | 8,064 |
| Gemma forget chess (300 neurons) | 345,600 | 0.027% | 345,600 |
| MIT edit on Qwen 0.8B (2 matrices) | 130,188 nonzero diffs | 0.016% | rank 1 on 64 columns per layer |

A rank-1 edit touches almost every entry of its matrix (changes below bf16 resolution don't count) but is defined
by two vectors.

## Known limits (stated in the write-ups, not fixed)
Single runs and small test sets: Newton 23 questions, mood 10 + 15 per setting, chess quiz 20 questions. The mood
judge is the unedited Gemma itself. Some edit settings were chosen after comparing on the same quiz (chess).
