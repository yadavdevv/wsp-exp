# Case 3 · Rules nobody has seen before (2026-11-15)

Is the writer (Claude) recalling famous constructions, or reasoning? Gemini, the referee, invented three new grid
rules in secret, outside this repo, and gave the writer only the plain-English specs (`spec1.md` to `spec3.md`).
The writer built a net for each from the spec alone, with no test runs; the nets were hashed (`committed.sha`)
and graded by the referee with the shared grader, [`../case02-game-of-life/ca.py`](../case02-game-of-life/ca.py).
The referee's instructions are in `GEMINI_RULEBOOK.md`.

## The rules, as the referee wrote them
- **R1** (2 states, B268/S4): a dead cell is born with exactly 2, 6 or 8 live neighbours; a live cell survives only
  with exactly 4.
- **R2** (2 states, side vs corner counts): a dead cell is born with at least 1 live side neighbour and exactly 1
  live corner; a live cell survives with 1-2 live sides and 1-2 live corners.
- **R3** (3 states, two colonies A and B): an empty cell joins A with exactly 2 A neighbours and no B (and vice
  versa); an A cell with 2 or more B neighbours converts to B, otherwise it survives with 2-3 A neighbours, else
  dies. B behaves the same way.

## Results
| Rule | Net | Params | First try |
|---|---|---|---|
| R1 | `net1.json` | 133 | **perfect** |
| R2 | `net2.json` | 161 | **perfect** |
| R3 | `net3.json` | 843 | **fail**: 94.8% of cells, 23.8% of boards |
| R3, after one round of feedback | `net3_v2.json` (`committed_v2.sha`) | 843 | perfect |

**First try 2/3, 3/3 after one fix.** Same grading as Case 2: 2,000 random boards one tick, plus 20 boards run for
200 ticks; pass = 100% of boards and 20/20 rollouts.

## The R3 failure
A coding bug, not a design error. The helper `gated(count_ch, centre_state)` built
`{count_ch: NB, centre_state: 10*CEN}`; for the survival terms both keys were the same channel, so the neighbour
kernel was silently overwritten. Every wrong cell was a survival case. The writer diagnosed it without the secret
answers: a literal Python implementation of `spec3.md` (`diag/my_reading_rule3.py`) disagreed with net 3 exactly
where the referee did, so the spec was unambiguous and the error was the writer's. The two-line fix
(`build_novel_v2.py`) leaves nets 1 and 2 byte-identical.

## Takeaway
The writer used the same toolkit as for the famous rules in Case 2 (exact integer indicators from ReLU hinges,
gated on the centre state, ANDed with `relu(a + b - 1)`): a **transferable method**, not memorised networks.
Caveat: the new rules are still neighbour-count rules on a grid, the same family. A task outside that family (e.g.
"is this 8-bit number divisible by 3?") is the obvious next test.
