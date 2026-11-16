# Referee rulebook — "Write a network for a rule nobody has seen"

You are the **referee** in a blind experiment. Another AI (the "writer") will try to hand-write neural-network
weights, with no training, that implement grid rules. You invent 3 **new** rules and keep them secret.
The writer must never see anything except the plain-English spec files described in §3.

## 1. Where to work
- Work ONLY in `~/Desktop/personal/wsp-referee/life/` (create it). Never write into `builder_mode/`.
- Do not read anything in `builder_mode/wsp/life/` except `ca.py` (copy it into your folder; it is the grader).

## 2. Invent 3 rules
Each rule is a cellular automaton on a wrapped (torus) grid, deterministic, next state of a cell depends only on
its own state and its neighbours. Make one of each kind:

| id | kind | requirements |
|---|---|---|
| R1 | **2 states, counting rule** | 2-D, 3x3 neighbourhood, next state depends only on (own state, number of live neighbours). Must NOT be any named rule (Conway B3/S23, HighLife B36/S23, Seeds B2/S, Day & Night B3678/S34678, Life without Death, Maze, Diamoeba, 2x2, Morley, Anneal, Replicator, etc.). Pick the birth/survival sets at random, then reject until it is unnamed AND "interesting" (see §4). |
| R2 | **2 states, position matters** | 2-D, neighbourhood at most 5x5. The rule must treat neighbours differently by position (e.g. weights by direction, only certain offsets count, an asymmetric pattern) so it is NOT a plain count of the 8 neighbours. Not a known rule from the literature. |
| R3 | **3 states** | 2-D, 3x3 neighbourhood, states 0, 1, 2. Not Brian's Brain, Wireworld, Greenberg-Hastings, Langton loops or any other named rule. |

Difficulty target: each rule should be describable in **at most 5 plain sentences**, about as hard to state as
Conway's Game of Life. Do not make them deliberately tricky or ambiguous; the test is reasoning, not puzzles.

## 3. Files to produce (per rule, N = 1, 2, 3)
- `ruleN.py` — **SECRET.** Defines `STATES = 2 or 3` and `step(board)` (numpy int array -> numpy int array,
  same shape, wrap-around edges via `np.roll`). This is the ground truth.
- `specN.md` — **given to the writer.** Contains ONLY:
  1. the rule in plain English (≤ 5 sentences, exact and unambiguous: say which offsets count, what happens in
     every case, and that edges wrap);
  2. the number of states;
  3. **two worked examples**: a small neighbourhood drawn as a grid and the resulting next state of the centre.
  Nothing else: no code, no hints about how to build a network, no test boards.

## 4. Self-checks before you hand anything over
For each rule run a 64x64 random board for 200 ticks and confirm:
- the board does not die out or freeze within 20 ticks, and does not fill to > 95% of one state;
- your `specN.md` alone determines the rule: re-implement it from the spec text in a fresh function and check it
  matches `ruleN.py` on 1,000 random boards. Fix the spec until it does.
Record what you checked in `referee_log.md`.

## 5. Grading (later, when the writer's nets arrive)
The user will copy the writer's `netN.json` files into your folder. For each, run
`python ca.py netN.json ruleN.py` and paste the full output back to the user. A net **passes** if
`one_step_boards_perfect == 1.0` and all 20 rollouts are perfect for 200 ticks.
Then (only then) also give the user `ruleN.py` contents so the writer can explain any failure.
