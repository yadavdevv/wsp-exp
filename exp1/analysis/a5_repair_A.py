# Round A repairs: hand edits derived from layer-1 geometry. Nothing is run on inputs.
import sys, json; sys.path.insert(0, "kit")
import numpy as np, ordergate as og
from evaluate import params_changed
OUT = "exp1/submission_A"
DIST, VAL, RAIN, HOUR = 0, 1, 4, 5

# C: distance units 5 and 13 ignore rain. Give them the rain weight every healthy-looking net uses:
# 0.20 x the distance weight = a 3 km earlier switch-on when raining (8 km -> 5 km).
c = og.load_net("public/net_C.json")
for j in (5, 13):
    c["W1"][RAIN, j] = 0.20 * c["W1"][DIST, j]
og.save_net(c, f"{OUT}/net_C_fixed.json")

# F: unit 6 switches on at 9.79 km dry / 4.73 km wet. Move the dry hinge to 7.87 km (bias) and set the
# rain weight back to 0.20 x distance weight so the wet hinge stays ~4.9 km.
f = og.load_net("public/net_F.json")
w = f["W1"][DIST, 6]
f["b1"][6] += w * (9.79 - 7.87) / 15
f["W1"][RAIN, 6] = 0.20 * w
og.save_net(f, f"{OUT}/net_F_fixed.json")

# D: unit 3 (the only strong value detector) has hour weight 0.19 x |w|, moving the R2 threshold from
# Rs 3 (midnight) to Rs 197 (11pm). Zero the hour weight and fold its value at hour 12 into the bias,
# keeping the threshold at ~Rs 104 at every hour.
d = og.load_net("public/net_D.json")
d["b1"][3] += d["W1"][HOUR, 3] * (12 / 23)
d["W1"][HOUR, 3] = 0.0
og.save_net(d, f"{OUT}/net_D_fixed.json")

for n, net in [("C", c), ("F", f), ("D", d)]:
    print(n, "params changed:", params_changed(og.load_net(f"{OUT}/net_{n}_fixed.json"), og.load_net(f"public/net_{n}.json")))

report = {
    "net_A": {"verdict": "healthy", "affects": [], "explanation": "Distance units switch on at ~7.9 km dry / ~4.9 km wet; value units at ~Rs 103-121, flat over the day; rider unit at r<1; no strong item/rating weights."},
    "net_B": {"verdict": "healthy", "affects": [], "explanation": "Same canonical structure (distance 7.6 dry / 4.7 wet, rider r<0.9, hour edge ~6); no strong item/rating dependence."},
    "net_C": {"verdict": "flawed", "affects": ["R5"], "explanation": "R5 (rain + distance > 5 km) is ignored. Both distance detectors (units 5, 13) have rain weight 0.00, so they switch on at 7.9 km whether or not it is raining; no other unit carries a meaningful rain signal. Rainy orders at 5-8 km are accepted. Fix: give units 5 and 13 rain weight 0.20 x distance weight (as in the other nets)."},
    "net_D": {"verdict": "flawed", "affects": ["R2"], "explanation": "The minimum-order rule depends on the hour. The only strong value detector (unit 3) has hour weight +0.19 x |w|, so the Rs 99 threshold becomes ~Rs 3 at midnight, ~Rs 54 at 6am, ~Rs 104 at noon, ~Rs 197 at 11pm: low-value orders accepted in the morning, valid Rs 99-197 orders rejected in the evening. Fix: zero that hour weight and fold its noon value into the bias."},
    "net_E": {"verdict": "healthy", "affects": [], "explanation": "Canonical structure (distance 7.9 dry / 4.9 wet, value ~Rs 130-137 flat, rider r<0.9, hour edges ~3-6.7)."},
    "net_F": {"verdict": "flawed", "affects": ["R1"], "explanation": "Dry distance limit moved from 8 km to ~10 km. The main distance detector (unit 6) switches on at 9.8 km dry but 4.7 km wet; its rain weight is 0.34 x distance weight vs 0.20 in healthy nets, i.e. the net kept the 5 km rain limit while the dry limit shifted ~2 km. Orders 8-10 km in dry weather are accepted. Fix: shift unit 6's bias so it switches on at 7.9 km dry and set rain weight to 0.20 x distance weight. Lower confidence than C/D; F also has unusual hour/value units with hour hinges near 15-20."},
}
json.dump(report, open(f"{OUT}/report.json", "w"), indent=2)
print("wrote", OUT)
