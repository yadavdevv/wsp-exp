# Fixing Gemma's decimal Yes/No bug with one rank-1 weight edit (2026-11-16)

## Question
Small models famously say 9.11 > 9.9. Can a small weight edit (no training) fix this kind of question?

## Model choice
- **Gemma 3 270M**: chance on everything (even 8.7 vs 12.1). It always answers "Yes" (160/160) and in "which is
  bigger" almost always names the second number (153/160). No comparison skill to repair, so not used.
- **Gemma 3 1B (instruction-tuned, bf16, 1.30B params)**: used.

## The bug (`decimals.py`, scored by comparing log-probs of the two answers; checked against real generation)
| question form | 1B accuracy |
|---|---|
| "Which number is bigger, a or b?" | 88-100% (trap pairs like 9.11 vs 9.9: 91%) |
| "Is a bigger than b? Answer Yes or No." | ~50-67% (says "Yes" ~79%) |

The same model that names the bigger number correctly cannot say Yes/No about it. (A similar Yes bias was reported
for Gemma-2-2B: divyanshsinghvi.github.io/blog/2025/debugging-numeric-comparisons-llms.)

## Where the answer is (`probe_layers.py`)
A linear probe on the residual stream at the last prompt token reads the true answer to the Yes/No question with
99.5% held-out accuracy from layer 9 to the last layer (26) (98% on trap pairs). **The model knows; only the final
readout to "Yes"/"No" is wrong.**

## The edit (`edit_yesno.py`, `fit_v2.py`)
One rank-1 change to layer 25 `mlp.down_proj` (1152 x 6912): `W' = W + u r^T`.
- `u` (write): the "Yes minus No" output direction, pulled back through the two RMSNorm gains (unit length).
- `r` (read, 6912 numbers): least squares on MLP-neuron activations so that
  - on 800 training comparisons ("bigger" and "smaller" phrasing, pairs seed 11) it moves the Yes-No margin to
    +-4 logits in the right direction (1 unit of read = 11.7 logits, measured, `calib.py`);
  - on 5,136 neutral activations (every token of 80 GSM8K questions, 98 general yes/no questions including
    non-number comparisons like "Is an elephant bigger than a mouse?") it does nothing.
- Size: 8,064 numbers define it; change / matrix = 0.12 (Frobenius).

Versions: v1 (probe on "bigger" only, strength hand-set) fixed "bigger" but broke "smaller" (31 -> 19%) and normal
text (perplexity 25 -> 224). v2 (both phrasings + neutral text) fixed that but broke 4 general comparison questions
("Is the Moon bigger than the Earth?"). **v3 (final)** adds non-number comparisons to the neutral set.

## Results (all on data not used for fitting; `eval_edit.py`, `check_gen.py`, `trap_grid.py`)

Audit (2026-11-17, `rescore_clean.py`): 10 of the 200 test pairs and 4 of the 360 grid questions shared a number pair
with the training pairs. With those removed: "bigger" 66.8 -> **98.9%** (n=190), traps 62.5 -> **100%**, "smaller"
32.6 -> **95.8%**, all trap questions 70.5 -> **93.0%** (n=356; truth No 42.1 -> 99.4%, truth Yes 98.9 -> 86.5%).
Use these numbers; the table below is the original run (differences are under 1 point).
| test | unedited | v3 edit |
|---|---|---|
| "Is a bigger than b?" (200 new pairs) | 67% | **98.5%** |
| same, trap pairs only (n=50) | 64% | **100%** |
| "Is a smaller than b?" (trained phrasing, new pairs) | 31.5% | **96%** |
| **all 360 trap questions** (x.d vs x.dd, x in 3/9/12, both orders) | 70.8% | **93.1%** |
| ... where truth is "No" | 42.8% | 99.4% |
| ... where truth is "Yes" | 98.9% | **86.7% (edit introduces errors)** |
| real generated answers, 80 new pairs | 54/80 | **79/80** |
| unseen phrasing "greater" / "larger" / "less" | 65 / 69 / 36% | 75 / 83 / 56% (partial) |
| whole numbers (n=9) / 3-decimal (n=12) / negatives (n=19) | 44 / 67 / 42% | 78 / 83 / 37% |
| "Which is bigger?" form (unchanged?) | 88.5% | 88.5% |
| 60 general-knowledge yes/no questions | 90% | 91.7% |
| perplexity, unrelated paragraph | 24.7 | 25.9 |
| control: same-size edit that only pushes toward "No" | 67% ("bigger") | 71.5% |

**Honest caveat:** the famous "Is 9.9 bigger than 9.11?" itself stays wrong after the edit (margin -1.25 -> -1.75).
So do 9.9 vs 9.12, 9.9 vs 9.15 and 12.9 vs 12.11, which the unedited model got right (pushed from Yes to No).

## Takeaways
1. The knowledge was already inside; the bug is in the readout. A rank-1 edit that reads the model's own signal
   fixes most of it (trap questions 71 -> 93%, generated answers 54 -> 79 of 80), with no damage to general questions.
2. It is not a "rule" fix: it generalises only partly to new phrasings, not to negative numbers, and over-corrects
   some true "Yes" cases (including the famous one).
3. The bias-only control does almost nothing, so the edit uses the comparison information, not just "say No more".
4. Side effects must be designed out explicitly: the first version wrecked normal text.
