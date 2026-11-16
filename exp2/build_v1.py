"""Hand-written tic-tac-toe net, v1. Every weight is set from the design below; nothing is fitted.

Input x: x[i] = 1 if cell i is mine, x[9+i] = 1 if theirs (mover's view). Score(c) is read for legal cells only.

Layer 1 (one unit per line L through cell c, with the line's other cells a, b):
  Win(L,c)   = relu(m_a + m_b - 1)               1 iff a, b both mine     -> playing c wins
  Block(L,c) = relu(t_a + t_b - 1)               1 iff a, b both theirs   -> c must be blocked
  OT(L,c)    = relu(t_a + t_b - 2m_a - 2m_b)     >=1 iff the line has theirs and none of mine
                                                  (their move at c would make two-in-a-row on L)
  T(L,c,e)   = relu(m_a - m_e - t_e)             1 iff a is mine and e is empty: playing c makes a
                                                  two-in-a-row on L and forces their reply at e
Layer 2 (per cell c):
  WinS, BlockS, TS = sums of the above over lines through c        (identity: inputs are >= 0)
  Fork(c)    = relu(sum T(.,c,.) - 1)            I get two threats at once
  OF(c)      = relu(sum OT(.,c) - 1)             they could fork at c
  Bad(L,c,e) = relu(2 T(L,c,e) + sum_{L' through e} OT(L',e) - 3)
               1 iff my threat forces them onto a cell where they fork (the X0-O4-X8 trap)
Output:
  score(c) = 1000 WinS + 200 BlockS + 60 Fork + 8 OF + 4 TS - 20 sum Bad(.,c,.) + pos(c)
  pos: centre 3, corners 2, sides 1.
"""
import os
import numpy as np
from ttt import LINES, save_net

M, T = 0, 9  # offsets of "mine" and "theirs" in the input
l1 = []      # (kind, line, c, e, weights{input: w}, bias)
for L in LINES:
    for c in L:
        a, b = [x for x in L if x != c]
        l1.append(("win", L, c, None, {M + a: 1, M + b: 1}, -1))
        l1.append(("block", L, c, None, {T + a: 1, T + b: 1}, -1))
        l1.append(("ot", L, c, None, {T + a: 1, T + b: 1, M + a: -2, M + b: -2}, 0))
        for e, other in ((a, b), (b, a)):
            l1.append(("t", L, c, e, {M + other: 1, M + e: -1, T + e: -1}, 0))

W1, b1 = np.zeros((18, len(l1))), np.zeros(len(l1))
for j, (_, _, _, _, w, bias) in enumerate(l1):
    for i, v in w.items():
        W1[i, j] = v
    b1[j] = bias


def units(kind, c=None, e=None, L=None):
    return [j for j, u in enumerate(l1) if u[0] == kind and (c is None or u[2] == c)
            and (e is None or u[3] == e) and (L is None or u[1] == L)]


l2 = []  # (kind, c, {l1 unit: w}, bias)
for c in range(9):
    l2.append(("winS", c, {j: 1 for j in units("win", c)}, 0))
    l2.append(("blockS", c, {j: 1 for j in units("block", c)}, 0))
    l2.append(("tS", c, {j: 1 for j in units("t", c)}, 0))
    l2.append(("fork", c, {j: 1 for j in units("t", c)}, -1))
    l2.append(("of", c, {j: 1 for j in units("ot", c)}, -1))
for j in units("t"):
    _, L, c, e, _, _ = l1[j]
    w = {k: 1 for k in units("ot", e)}
    w[j] = 2
    l2.append(("bad", c, w, -3))

W2, b2 = np.zeros((len(l1), len(l2))), np.zeros(len(l2))
for k, (_, _, w, bias) in enumerate(l2):
    for j, v in w.items():
        W2[j, k] = v
    b2[k] = bias

OUT_W = {"winS": 1000, "blockS": 200, "fork": 60, "of": 8, "tS": 4, "bad": -20}
POS = [2, 1, 2, 1, 3, 1, 2, 1, 2]
W3, b3 = np.zeros((len(l2), 9)), np.array(POS, dtype=float)
for k, (kind, c, _, _) in enumerate(l2):
    W3[k, c] = OUT_W[kind]

os.makedirs("handwritten", exist_ok=True)
save_net([(W1, b1), (W2, b2), (W3, b3)], "handwritten/ttt_v1.json")
nz = int(sum((W != 0).sum() + (b != 0).sum() for W, b in [(W1, b1), (W2, b2), (W3, b3)]))
print(f"layers 18 -> {len(l1)} -> {len(l2)} -> 9, nonzero params {nz}")
