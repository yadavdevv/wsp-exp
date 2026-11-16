# Round B repairs: hand edits computed from the measured boundaries, then checked by probing.
import sys; sys.path.insert(0, "kit"); sys.path.insert(0, "exp1/analysis")
import numpy as np, ordergate as og
from b1_map import boundary
DIST, VAL, RAIN = 0, 1, 4
out = {}

# D: measured R2 boundary ~Rs 49 instead of Rs 99. Unit 3 is the value detector (w_val = -8.20).
# Move its switch-on point up by Rs 50: bias += |w| * 50/1000. Hour weight left alone (it is compensated).
d = og.load_net("public/net_D.json")
d["b1"][3] += -d["W1"][VAL, 3] * 50 / 1000
out["D"] = d

# F: measured dry boundary 10.05 km. Round A moved unit 6 by 1.92 km and got ~9.05, i.e. ~1 km of
# boundary per 1 km of hinge shift is lost to other units -> shift unit 6 by the full 2.05 km plus the
# 1 km shortfall, and keep its wet switch-on point fixed by lowering the rain weight by the same amount.
f = og.load_net("public/net_F.json")
w = f["W1"][DIST, 6]
shift = (10.05 - 8.0) + 1.0
f["b1"][6] += w * shift / 15
f["W1"][RAIN, 6] -= w * shift / 15
out["F"] = f

for n, net in out.items():
    og.save_net(net, f"exp1/analysis/tmp_{n}.json")
D, F = (og.load_net(f"exp1/analysis/tmp_{n}.json") for n in "DF")
print("D min order by hour:", {h: boundary(D, "order_value", 0, 400, 1, hour=h) for h in [0, 6, 12, 18, 23]})
print("F dry by hour:", {h: boundary(F, "distance_km", 0, 12, .05, hour=h) for h in [0, 6, 12, 18, 23]},
      "wet:", boundary(F, "distance_km", 0, 12, .05, raining=1))
