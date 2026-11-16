"""Hand-written nets for the two other famous rules, written zero-shot (no test run before saving).

Rule 110 (1-D, board height 1; neighbourhood l, c, r).  New cell = 1 except for patterns 000, 100, 111.
  Exact-pattern detectors with ReLU (a pattern with k ones scores k only on an exact match):
    [000] = relu(1 - l - c - r)    [100] = relu(l - c - r)    [111] = relu(l + c + r - 2)
  score = 1 - [000] - [100] - [111];  next = score > 0.5.
  Only the middle row of each 3x3 kernel is used (height-1 board wraps onto itself vertically).

Brian's Brain (states 0 off, 1 on, 2 dying; one-hot input channels x0, x1, x2).
  on -> dying; dying -> off; off -> on iff exactly 2 of the 8 neighbours are on.
  s = (# on neighbours) + 10*x0_centre - 10   (= n when the cell is off, <= -2 otherwise)
  [n == 2 and off] = relu(s - 1) - 2 relu(s - 2) + relu(s - 3)   (1 at s = 2, 0 at every other integer)
  outputs (argmax): off = 1,  on = 2*[born],  dying = 2*relu(x1_centre)
"""
import json
import numpy as np


def k3():
    return np.zeros((3, 3))


# Rule 110: 1 input channel, 3 hidden, 1 output
W = np.zeros((3, 1, 3, 3))
W[0, 0, 1] = [-1, -1, -1]
W[1, 0, 1] = [1, -1, -1]
W[2, 0, 1] = [1, 1, 1]
rule110 = {"states": 2, "layers": [
    {"k": 3, "W": W.tolist(), "b": [1, 0, -2]},
    {"k": 1, "W": [[[[-1]], [[-1]], [[-1]]]], "b": [1]}],
    "output": {"type": "threshold", "t": 0.5}}

# Brian's Brain: 3 input channels, 4 hidden, 3 outputs
W = np.zeros((4, 3, 3, 3))
for h, bias in enumerate([-11, -12, -13]):
    W[h, 1] = 1
    W[h, 1, 1, 1] = 0
    W[h, 0, 1, 1] = 10
W[3, 1, 1, 1] = 1
W2 = np.zeros((3, 4, 1, 1))
W2[1, :3, 0, 0] = [2, -4, 2]
W2[2, 3, 0, 0] = 2
brian = {"states": 3, "layers": [
    {"k": 3, "W": W.tolist(), "b": [-11, -12, -13, 0]},
    {"k": 1, "W": W2.tolist(), "b": [1, 0, 0]}],
    "output": {"type": "argmax"}}

json.dump(rule110, open("nets/rule110_hand.json", "w"))
json.dump(brian, open("nets/brian_hand.json", "w"))
print("saved")
