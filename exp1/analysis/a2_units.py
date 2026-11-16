# Round A: dump layer-1 units with weights normalized by the unit's largest |w|, and the signed
# downstream sign estimate s_j = W2[j] . W3 (linear path, ignores layer-2 gating).
import sys; sys.path.insert(0, "kit")
import numpy as np, ordergate as og
np.set_printoptions(precision=2, suppress=True, linewidth=160)
short = ["dist", "val", "open", "ride", "rain", "hour", "item", "rate"]
for n in sys.argv[1]:
    net = og.load_net(f"public/net_{n}.json")
    W1, b1 = net["W1"], net["b1"]
    s = (net["W2"] @ net["W3"])[:, 0]
    print(f"\n=== net_{n}  b2={np.round(net['b3'],2)}")
    print("unit " + "".join(f"{x:>7}" for x in short) + f"{'bias':>7}{'|w|max':>8}{'down':>7}")
    for j in np.argsort(-np.abs(W1).max(0) * np.abs(s)):
        m = np.abs(W1[:, j]).max()
        print(f"{j:>4} " + "".join(f"{w:>7.2f}" for w in W1[:, j] / m) + f"{b1[j]/m:>7.2f}{m:>8.2f}{s[j]:>7.2f}")
