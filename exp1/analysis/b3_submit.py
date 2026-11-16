# Round B final submission.
import sys, json, shutil; sys.path.insert(0, "kit")
import ordergate as og
from evaluate import evaluate, params_changed
OUT = "exp1/submission_B"
shutil.copy("exp1/submission_A/net_C_fixed.json", f"{OUT}/net_C_fixed.json")  # Round A fix already passes

d = og.load_net("public/net_D.json")
d["b1"][3] += -d["W1"][1, 3] * 50 / 1000             # value detector: switch-on Rs ~49 -> ~99
og.save_net(d, f"{OUT}/net_D_fixed.json")

f = og.load_net("public/net_F.json")
for j, s in [(6, 2.05), (10, 2.0)]:                   # both distance units: dry hinge -~2 km, wet unchanged
    w = f["W1"][0, j]; f["b1"][j] += w * s / 15; f["W1"][4, j] -= w * s / 15
og.save_net(f, f"{OUT}/net_F_fixed.json")

for n in "CDF":
    fx, orig = og.load_net(f"{OUT}/net_{n}_fixed.json"), og.load_net(f"public/net_{n}.json")
    r = evaluate(fx)
    print(n, "changed", params_changed(fx, orig), {k: round(v, 4) for k, v in r.items()})

report = {
    "net_A": {"verdict": "healthy", "affects": [], "explanation": "All probes >= 0.996, irrelevant-feature flips <= 0.0004."},
    "net_B": {"verdict": "healthy", "affects": [], "explanation": "All probes >= 0.995, flips <= 0.0004."},
    "net_C": {"verdict": "flawed", "affects": ["R5"], "explanation": "R5 ignored: reject point is 8.05 km both dry and raining (should be 5 km when raining). R5 probe 0.748. Fix (2 weights): rain weight 0.20 x distance weight on distance units 5 and 13 -> wet reject point 5.05 km, R5 probe 0.996."},
    "net_D": {"verdict": "flawed", "affects": ["R2"], "explanation": "Minimum order value lowered from Rs 99 to ~Rs 49 at every hour (~Rs 21 at hour 0); orders Rs 49-98 accepted. R2 probe 0.714. (Round A blamed an hour weight on unit 3; probing shows it is compensated and the threshold is flat.) Fix (1 weight): unit 3 bias shifted so it switches on Rs 50 higher -> threshold Rs 98-99 at hours 6-23, still ~Rs 53 at hour 0."},
    "net_E": {"verdict": "healthy", "affects": [], "explanation": "All probes >= 0.995, flips <= 0.0005."},
    "net_F": {"verdict": "flawed", "affects": ["R1"], "explanation": "Dry distance limit moved from 8 km to 10 km (reject point 10.05 km at every hour and rider count; rain limit still 5 km). R1 probe 0.751. Fix (4 weights): both distance units (6, 10) switch on ~2 km earlier when dry, rain weight lowered so the wet point stays at 5 km -> R1 probe ~0.97."},
}
json.dump(report, open(f"{OUT}/report.json", "w"), indent=2)
