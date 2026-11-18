# Case 5 · Fixing a lunar lander pilot by editing its weights (2026-11-15)

## Setup
- `lander.py`: 2-D lander (gravity 1.6, main engine 4.0, side thrusters 0.6 lateral / 3.0 torque). Success =
  touchdown with |x| < 1, |vy| < 1, |vx| < 0.5, |tilt| < 0.25, |spin| < 0.5. Feasibility check: a hand PD
  controller lands 97.7% in calm air.
- Pilot: MLP 7 -> 64 -> 64 -> 3 (tanh), **4,867 weights**, trained with OpenAI-ES (antithetic, Adam) for 1,500
  generations in **calm air only** (`train_es.py`, `pilots/calm_final.json`): 100% landings, fuel 3.65 (PD: 4.24).
- Inputs include a wind sensor that read 0 throughout training.
- Rules for hand patches: design from weights, physics and calm-air probing only; save + hash
  (`pilots/patches_committed.sha`) before any flight in the failure condition; each revision reported.
- Baseline: ES fine-tune from the calm pilot, 300 generations in the failure condition.
- All numbers: the same 2,000 random starts (seed 2024).

## Results
| Case | Original | Hand patch | Weights changed | Attempts | Retrained (ES) | Weights changed | Retrained, calm air |
|---|---|---|---|---|---|---|---|
| a. gusty crosswind (0.3) | 0% | **100%** | 64 | 1 | 100% | 4,867 | 100% |
| b. left thruster stuck +0.9 | 49% | **91%** | 258 | 2 (v1 64%) | 100% | 4,867 | **61%** |
| c. cargo x2.2 | 4% | **92%** | 259 | 2 (v1 45%) | 100% | 4,867 | 100% |
| d. altimeter +3 m | 0% | **100%** | 64 | 1 | 100% | 4,867 | 100% |

## The patches
- **a.** The never-trained wind-sensor weights had norm 5.2, as large as the real inputs: a random signal waiting
  for the first wind. Patch: replace that column with the x column times k/5, so wind acts as a virtual
  position offset. k = 1/(hover_main · P_MAIN · slope), where slope = 0.322 rad/m is the pilot's equilibrium tilt
  per metre of perceived offset (measured by static probing) → k = 1.97 m per unit wind. Because it reads the real
  sensor, it generalises to gusts up to 0.4 (the original fails from 0.2).
- **b.** v1: left output forced to ~0; right output refit (least squares on calm-flight activations, ROME-style) to
  0.9 + right − left. 64%: it replays the old steering, swings the craft (|tilt| 0.48), saturates the right
  thruster 25% of the time, and leaves a net clockwise push (+0.084) → tumble. The retrained pilot instead stays
  upright (|tilt| 0.15). v2 adds a perception change: tilt and spin read 2× larger (2 columns) → 91%.
- **c.** v1: all outputs refit to 2.2× (force/mass then matches calm, since gravity is per unit mass). 45%:
  every failure was only a too-fast touchdown (median vy −1.33); the engine saturates and braking is ~11× weaker.
  v2: vertical speed perceived 1.5× larger → the pilot's usual descent is 1.5× slower in reality → 92%.
- **d.** Exact: `b1 -= W1[:, y] * 0.3` (64 biases).

## Takeaways
1. Where the fix is "correct a signal" (a, d), the weight edit is exact, tiny (1.3% of weights) and first-try.
2. Where the fault demands a new strategy (b, c), replaying the old behaviour fails; a second, physics-guided edit
   to the pilot's *perception* recovers most of the gap (91–92% vs 100% retrained).
3. Retraining always reaches 100% but rewrites every weight, and in case b it forgot calm flight (61%).
4. Hand patches are fault-specific: patch b lands **0%** of the time with no fault (a, c, d: 100%). Deploy only
   after diagnosis.

## Visuals (`make_visuals.py`, `visuals/`)
`{a_gust,b_stuck,c_cargo,d_altimeter}.gif`: original vs hand patch vs retrained, on the first test start where
the original fails (not hand-picked). `*_heatmap.png`: success vs failure strength × start position.
