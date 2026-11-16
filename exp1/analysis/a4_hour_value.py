# Round A: every unit with a meaningful hour or value weight; hinge along that feature at REF,
# plus how the value hinge moves between hour 0, 6 and 23 (layer-1 algebra only).
import sys; sys.path.insert(0, "kit")
import numpy as np, ordergate as og
REF = np.array([3, 400, 1, 5, 0, 12, 8, 3.0])
for n in "ABCDEF":
    net = og.load_net(f"public/net_{n}.json")
    W1, b1 = net["W1"], net["b1"]
    down = (np.abs(net["W2"]) @ np.abs(net["W3"]))[:, 0]
    print(f"--- net_{n}")
    for j in range(16):
        for f, name in [(5, "hour"), (1, "val")]:
            if abs(W1[f, j]) < 1.0 or abs(W1[f, j]) < 0.5 * np.abs(W1[:, j]).max():
                continue
            line = f"   u{j:<2} {name:<4} w={W1[f,j]:6.2f} down={down[j]:5.0f}"
            for h in ([0, 6, 12, 23] if f == 1 else [None]):
                x = og.normalize(REF[None])[0].copy()
                if h is not None: x[5] = h / 23
                rest = W1[:, j] @ x - W1[f, j] * x[f] + b1[j]
                raw = -rest / W1[f, j] * og.SCALE[f] + og.OFFSET[f]
                line += f"  {'>' if W1[f,j] > 0 else '<'}{raw:7.1f}" + (f"@h{h}" if h is not None else "")
            print(line)
