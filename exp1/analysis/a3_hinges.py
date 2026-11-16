# Round A: layer-1 hyperplane geometry only. For each strong unit, where does it switch on along
# its dominant feature, with the other features held at a reference passing order?
# (Solves w.x + b = 0 for one coordinate; no layer-2/3 computation, no network output.)
import sys; sys.path.insert(0, "kit")
import numpy as np, ordergate as og
REF = np.array([3, 400, 1, 5, 0, 12, 8, 3.0])  # raw reference order (passes every rule)
short = ["dist", "val", "open", "ride", "rain", "hour", "item", "rate"]
for n in "ABCDEF":
    net = og.load_net(f"public/net_{n}.json")
    W1, b1 = net["W1"], net["b1"]
    infl = np.abs(W1).max(0) * (np.abs(net["W2"]) @ np.abs(net["W3"]))[:, 0]
    xr = og.normalize(REF[None])[0]
    rows = []
    for j in np.argsort(-infl)[:10]:
        f = int(np.argmax(np.abs(W1[:, j])))
        rest = W1[:, j] @ xr - W1[f, j] * xr[f] + b1[j]
        x_hinge = -rest / W1[f, j]                       # normalized hinge on feature f
        raw = x_hinge * og.SCALE[f] + og.OFFSET[f]
        side = ">" if W1[f, j] > 0 else "<"
        extra = ""
        if f == 0:  # distance unit: also hinge when raining
            raw_rain = -(rest + W1[4, j]) / W1[0, j] * og.SCALE[0]
            extra = f"  (raining: {side}{raw_rain:.2f})"
        rows.append(f"u{j:<2} {short[f]}{side}{raw:8.2f}{extra}  infl {infl[j]:.0f}")
    print(f"--- net_{n}"); print("\n".join("   " + r for r in rows))
