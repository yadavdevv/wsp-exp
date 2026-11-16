# Round A: weights-only. Path-norm influence of each input on the output (no forward pass).
import sys; sys.path.insert(0, "kit")
import numpy as np, ordergate as og
np.set_printoptions(precision=3, suppress=True, linewidth=150)
print(f"{'net':<6}" + "".join(f"{f[:10]:>12}" for f in og.FEATURES))
for n in "ABCDEF":
    net = og.load_net(f"public/net_{n}.json")
    # sum over paths |W1[i,j] W2[j,k] W3[k]|, then express per unit of raw feature range (inputs already in [0,1])
    P = np.abs(net["W1"]) @ np.abs(net["W2"]) @ np.abs(net["W3"])
    P = P[:, 0] / P[:, 0].sum()
    print(f"net_{n:<2}" + "".join(f"{p:>12.3f}" for p in P))
