# Case 6 · Fixing a walking robot by editing its weights, where it stops working (2026-11-15)

## Setup
- MuJoCo `Walker2d-v5` (gymnasium 1.4, mujoco 4.0): 17 sensors, 6 motors (hip/knee/ankle x 2), up to 1,000 steps.
- Policies: MLP with tanh, observation normalisation frozen in. Trained with OpenAI-ES on 6 worker processes
  (`train_es.py`, alive-bonus shift as in ARS), healthy robot only.
  | | Shape | Weights | Training | Healthy (30 test episodes) |
  |---|---|---|---|---|
  | small | 17-64-64-6 | 5,702 | 1,000 gens x 96 eps (~15 min) | 34.8 m, 0% falls |
  | big | 17-256-256-6 | 71,942 | 1,500 gens x 128 eps (~36 min) | 28.5 m, 7% falls |
- Faults (`walker.py`): a. R knee motor x0.5, b. R hip motor x0.6, c. torso mass x1.3, d. floor friction x0.7.
- Test: 30 episodes, seeds 50,000+ (same for every policy). Distance = metres walked; falls = episode ended by falling.
- Three repair methods:
  1. **Blind patch** (`patches.py`): designed from weights + physics + healthy-only probing, hashed before any
     faulty run. Same recipe for both sizes.
  2. **Budget patch** (`budget_patch.py`): 5 pre-registered candidate edits per fault, 4 practice episodes each
     (seeds 60,000+) = **20 faulty episodes**; best one tested once. (cf. MIT's 10/25/50 forward-pass budgets)
  3. **Retraining**: ES fine-tune from the healthy policy in the fault, 200 generations (~19k / ~26k episodes).

## How the trained walkers drive their motors (healthy)
Share of steps with |command| > 0.95: small 51-81% for hips/ankles, 5-18% knees; big **69-97% for every motor**.
Both are near bang-bang controllers; the big one more so. Little headroom for "push harder" fixes.

## Results (distance m / falls)
| Fault | | Original | Blind patch | Budget patch (20 eps) | Retrained | Retrained on healthy robot |
|---|---|---|---|---|---|---|
| a. weak knee | small | 9.6 / 90% | 18.5 / 50% | **28.9 / 10%** | 35.6 / 3% | 5.0 / 100% |
| | big | 3.2 / 100% | 3.7 / 100% | 3.8 / 100% | 26.4 / 3% | 1.7 / 100% |
| b. weak hip | small | 18.5 / 80% | 21.6 / 57% | 21.6 / 57% | 34.8 / 0% | 11.0 / 80% |
| | big | 4.1 / 100% | 4.7 / 100% | 5.2 / 100% | 26.8 / 0% | 16.5 / 50% |
| c. heavy torso | small | 5.3 / 100% | 4.2 / 100% | 15.5 / 80% | 32.7 / 0% | 24.6 / 50% |
| | big | 22.0 / 33% | 5.5 / 100% | **26.5 / 17%** | 29.6 / 0% | 24.6 / 20% |
| d. icy floor | small | 5.3 / 100% | 1.5 / 100% | 6.7 / 100% | 31.7 / 3% | 4.2 / 100% |
| | big | 8.3 / 100% | 1.9 / 100% | 12.5 / 63% | 26.7 / 10% | 17.5 / 60% |

Budget winners: small a = knee x1.5 (not the physics guess x2); small c, d and big d = "perceive forward speed
1.3x" (walks more carefully); big c = both hips x1.2. Small b: the candidate that looked perfect in 4 practice
episodes (32.9 m, 0% falls) fell 57% in the test: 4 episodes is too few to choose reliably.

## Takeaways
1. **Contrast with the lander** (`../case05-lunar-lander/RESULTS.md`): there, blind weight patches matched retraining because
   each fix was a known signal correction. A walking gait is a finely balanced, saturated rhythm; most faults need
   a new gait, and the weights don't say what it is. Blind patches mostly fail.
2. **A tiny experiment budget helps where a good candidate exists** (small knee: 50% -> 10% falls; big torso:
   33% -> 17%), at ~1/1000 of retraining's episodes. It cannot invent a new gait (ice, weak hip).
3. **The bigger network was harder to patch, not easier**: training pushed it further into saturation.
4. **Retraining always works but produces a specialist**: every retrained walker got worse on the healthy robot,
   up to 100% falls. Neither method gives robustness; that needs training on varied conditions.
5. Recurring trick across lander and walker: editing **perception** (input columns) rather than actions, e.g.
   "feel faster than you are", changes the strategy with a handful of weights.

## Visuals (`visuals/`, tracking camera)
`walker_{a_knee,c_torso,d_ice}.gif` (small) and `walker256_*.gif` (big): original vs budget patch vs retrained, on
the first test seed where the original falls. `healthy.gif`: the trained gait.
