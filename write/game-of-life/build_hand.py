"""Hand-written Game of Life net. Every number is set from the design below; nothing is fitted or tried.

Let n = live neighbours (0-8) and c = the cell itself. Score s = n + c/2.
  alive next  <=>  n == 3, or (n == 2 and c == 1)  <=>  s in {2.5, 3, 3.5}
s only takes multiples of 0.5, so a bump that is 1 on [2.5, 3.5] and 0 at 2 and 4 is enough:
  bump(s) = 2 * (relu(s - 2) - relu(s - 2.5) - relu(s - 3.5) + relu(s - 4))
    s:     <=2   2.5   3   3.5   >=4
    bump:   0     1    1    1     0
Layer 1 = four copies of the same 3x3 kernel (neighbours 1, centre 0.5) with biases -2, -2.5, -3.5, -4.
Layer 2 = weights 2, -2, -2, 2. Output is exactly 0 or 1; threshold 0.5.
"""
import numpy as np
from life import save_net, n_params

kernel = np.ones((3, 3))
kernel[1, 1] = 0.5
net = {"W1": np.stack([kernel] * 4), "b1": np.array([-2, -2.5, -3.5, -4]),
       "w2": np.array([2, -2, -2, 2]), "b2": 0.0, "threshold": 0.5}
save_net(net, "nets/hand_v1.json")
print("params", n_params(net))
