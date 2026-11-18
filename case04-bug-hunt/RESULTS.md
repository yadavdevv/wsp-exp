# Experiment 1 results — bug hunt (graded by Gemini 3.8 Flash referee, 2026-11-14)

Planted (data poisoning): C = R5 ignored, D = R2 threshold 99 -> 49, F = R1 threshold 8 -> 10 km. A, B, E healthy.

| Net | Round A (weights only) | Round B (probing) | Fine-tune baseline |
|---|---|---|---|
| C | right rule + mechanism; fix 2 params, worst probe 0.756 -> 0.994 | same fix | 407 params, worst probe 0.984 |
| D | right rule, WRONG mechanism (blamed hour weight); fix 2 params, no gain (0.706) | right; fix 1 param -> 0.984 | 377 params, 0.993 |
| F | right rule + mechanism; fix 2 params -> 0.888 | fix 4 params -> 0.972 | 372 params, 0.993 |
| A/B/E | healthy, correct | healthy, correct | — |

Probe numbers use the grader's seed (999); `kit/evaluate.py` with that seed reproduces every value above
(re-checked 2026-11-17).

Verdicts: 6/6 both rounds (grader counts rule ids). Fully correct explanations: Round A 5/6, Round B 6/6.
Black-box probes found all three flaws in one ~2 s run; weight reading took several analysis passes
but located the 1-4 numbers to edit. Fine-tune is slightly more accurate but rewrites ~90% of weights.
Caveat: Gemini chose the easiest flaw kinds (ignored rule, two threshold shifts); no confounds or leaks.
