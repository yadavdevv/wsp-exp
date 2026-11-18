# Case 2 · The Game of Life, written by hand (2026-11-15)

Question: when a frontier LLM (here: Claude, the "writer") writes network weights with no training, can it compile
famous grid rules exactly? Every net was saved and hashed before any test run. The same method on rules nobody has
seen before (invented in secret by Gemini, the referee) is [Case 3](../case03-secret-rules).

## Harness
- `life.py` (Conway-specific) and `ca.py` (general CA net format + grader; the referee used this file).
- Grade: 2,000 random boards one tick (cell accuracy, share of boards fully correct) + 20 random boards run
  200 ticks (Conway: also glider, R-pentomino, Gosper gun for 1,000 ticks). Pass = 100% boards + 20/20 rollouts.
- Harness checks: glider moves (1,1) per 4 ticks, blinker period 2, Gosper gun +5 cells per 30 ticks, Rule 110
  draws its left-growing triangle; a net with one bias changed (-3.5 -> -3.0) fails at tick 1.

## Results
Grade any net: `../.venv/bin/python ca.py nets/brian_hand.json rules/brians_brain.py` (Conway: `life.py`).

| Rule | Famous? | Net | Params | Zero-shot result |
|---|---|---|---|---|
| Conway's Life (B3/S23) | yes | `nets/hand_v1.json` (`build_hand.py`) | 45 | perfect |
| Conway, Springer & Kenyon's minimal shape | yes | `nets/conway_hand_min.json` | 25 | perfect |
| Rule 110 (1-D) | yes | `nets/rule110_hand.json` | 34 | perfect |
| Brian's Brain (3 states) | yes | `nets/brian_hand.json` | 127 | perfect |

**Zero-shot: famous 4/4.** The secret new rules scored 2/3, then 3/3 after one fix ([Case 3](../case03-secret-rules)).

Method used for all of them, famous and secret: exact integer indicators built from ReLU hinges (`[x == k]`, `[x >= 1]`,
`[x in {a, a+1}]`), gated on the centre state with a +-10 offset, ANDed with `relu(a + b - 1)`. Same toolkit for
famous and new rules -> the evidence points to **transfer of a method**, not recall of specific nets. Caveat:
the new rules are still neighbourhood-count rules on a grid, i.e. the same family.

## Training baseline (Conway only; `train.py`, Adam, BCE, 20k steps, fresh random boards)
| Shape | Params | Trained runs perfect |
|---|---|---|
| minimal (Springer & Kenyon): 3x3 conv 2ch -> 1x1 -> 1x1 | 25 | **2 / 20** |
| 3x3 conv 4ch -> 1x1 (same as hand_v1) | 45 | 6 / 10 |
| 8 channels | 81 | 8 / 10 |
| 16 channels | 161 | 10 / 10 |
| 32 channels | 321 | 10 / 10 |

15 of the 18 failed minimal runs collapsed to "always predict dead" (84.138% accuracy = exactly the dead share
of the test boards; predicts 0 live cells). Matches Springer & Kenyon 2020 (arXiv 2009.01398): small nets
rarely learn Life; overcomplete ones do. The hand-written 25-param net is perfect.

## Takeaways
1. At the minimal size, a hand-written net is exact while training mostly fails (2/20); with a bit of extra
   width, training catches up — hand-writing is impressive mainly at the minimum.
2. Famous vs new made no difference in success rate: the writer reused one method.
3. The one failure is an engineer's bug; one round of feedback fixed it (cf. MIT's forward-pass budgets).
4. Next, if wanted: a task outside the grid-rule family (e.g. "8-bit input divisible by 3") to test reasoning
   beyond method transfer.
