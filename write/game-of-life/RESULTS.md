# Writing networks by hand: famous rules vs secret new rules (2026-11-15)

Question: when a frontier LLM (here: Claude, the "writer") writes network weights with no training, is it
recalling famous constructions or reasoning? Test: 4 famous grid rules vs 3 new rules invented blind by a
different model (Gemini, the referee), matched in kind and difficulty. Every writer net was saved and hashed
before any test run; secret rules were graded only by the referee.

## Harness
- `life.py` (Conway-specific) and `ca.py` (general CA net format + grader; the referee used this file).
- Grade: 2,000 random boards one tick (cell accuracy, share of boards fully correct) + 20 random boards run
  200 ticks (Conway: also glider, R-pentomino, Gosper gun for 1,000 ticks). Pass = 100% boards + 20/20 rollouts.
- Harness checks: glider moves (1,1) per 4 ticks, blinker period 2, Gosper gun +5 cells per 30 ticks, Rule 110
  draws its left-growing triangle; a net with one bias changed (-3.5 -> -3.0) fails at tick 1.

## Results
| Rule | Famous? | Net | Params | Zero-shot result |
|---|---|---|---|---|
| Conway's Life (B3/S23) | yes | `nets/hand_v1.json` (`build_hand.py`) | 45 | perfect |
| Conway, Springer & Kenyon's minimal shape | yes | `nets/conway_hand_min.json` | 25 | perfect |
| Rule 110 (1-D) | yes | `nets/rule110_hand.json` | 34 | perfect |
| Brian's Brain (3 states) | yes | `nets/brian_hand.json` | 127 | perfect |
| Secret R1: B268/S4 counting rule | no | `novel/net1.json` | 133 | **perfect** (referee) |
| Secret R2: orthogonal vs diagonal counts | no | `novel/net2.json` | 161 | **perfect** (referee) |
| Secret R3: 3 competing states | no | `novel/net3.json` | 843 | **fail**: 94.8% cells, 23.8% boards |
| Secret R3, v2 after feedback | no | `novel/net3_v2.json` | 843 | perfect (referee) |

**Zero-shot: famous 4/4, secret new 2/3.** The one failure was a coding bug, not a design error:
`gated(count_ch, centre_state)` built `{count_ch: NB, centre_state: 10*CEN}`; for the survival terms both keys
were the same channel, so the neighbour kernel was silently overwritten. Every wrong cell was a survival case.
Diagnosis was done without the secret rule: net3 scored identically against a literal implementation of
spec3 (`novel/diag/`), proving the spec was unambiguous and the error was the writer's. The 2-line fix leaves
net1/net2 byte-identical.

Method used for all 7: exact integer indicators built from ReLU hinges (`[x == k]`, `[x >= 1]`,
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
