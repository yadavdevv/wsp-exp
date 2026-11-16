"""Hand-written Life net in Springer & Kenyon's minimal shape (25 params), written zero-shot.

Shape: 3x3 conv (2 ch) -> ReLU -> 1x1 conv (1 ch) -> ReLU -> 1x1 conv (1 ch), threshold.
s = n + c/2 as in build_hand.py; alive next <=> s in {2.5, 3, 3.5}.
  h1 = relu(s - 2),  h2 = relu(s - 3.5)
  z  = relu(2 h1 - 8 h2 - 0.5)
    s:   <=2   2.5   3    3.5   4     >=4.5
    z:    0    0.5   1.5  2.5   0     0        (s = 4: 4 - 4 - 0.5 < 0; slope 2 - 8 < 0 after that)
  out = z, threshold 0.25.
"""
import json
import numpy as np

kernel = np.ones((3, 3))
kernel[1, 1] = 0.5
net = {"states": 2, "layers": [
    {"k": 3, "W": [[kernel.tolist()], [kernel.tolist()]], "b": [-2, -3.5]},
    {"k": 1, "W": [[[[2]], [[-8]]]], "b": [-0.5]},
    {"k": 1, "W": [[[[1]]]], "b": [0]}],
    "output": {"type": "threshold", "t": 0.25}}
json.dump(net, open("nets/conway_hand_min.json", "w"))
print("saved")
