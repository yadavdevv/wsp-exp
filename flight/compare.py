"""Same 2,000 starts for every pilot: original vs hand patch vs ES fine-tune, in the failure case and in calm air."""
import numpy as np
import lander as L

base = L.load_net("pilots/calm_final.json")
CASES = {"a_gust": ({"gust": 0.3}, "patch_a_gust"), "b_stuck": ({"left_stuck": 0.9}, "patch_b_stuck_v2"),
         "c_cargo": ({"mass_mult": 2.2}, "patch_c_cargo_v2"), "d_altimeter": ({"alt_bias": 3.0}, "patch_d_altimeter")}


def changed(p):
    return int(sum((np.abs(Wn - Wo) > 1e-6).sum() + (np.abs(bn - bo) > 1e-6).sum() for (Wn, bn), (Wo, bo) in zip(p, base)))


if __name__ == "__main__":
    print(f"{'case':12} | {'original':>8} | {'patch':>6} {'changed':>8} {'calm':>6} | {'finetune':>8} {'changed':>8} {'calm':>6}")
    for k, (c, pf) in CASES.items():
        p, f = L.load_net(f"pilots/{pf}.json"), L.load_net(f"pilots/ft_{k}.json")
        ev = lambda net, cc: L.evaluate(net, cc, n=2000, seed=2024)["success"]
        print(f"{k:12} | {ev(base, c):>8.1%} | {ev(p, c):>6.1%} {changed(p):>8} {ev(p, {}):>6.1%} | "
              f"{ev(f, c):>8.1%} {changed(f):>8} {ev(f, {}):>6.1%}")
