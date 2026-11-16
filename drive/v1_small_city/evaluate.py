"""Score a driver (network JSON or the expert) on held-out routes, rule by rule.

    python evaluate.py nets/driver.json [--side 1] [--n 40]       python evaluate.py expert

Per episode: route completed, collision, off-road, red-light runs; per step: out of lane (|offset from own lane
centre| > 1.2 m outside junctions), speeding (> limit + 1 m/s), wrong signal (should signal a turn within 30 m of
the junction / inside it, and not otherwise).
"""
import argparse, json
import numpy as np
from city import World, LANE

TEST_SEED0 = 500_000


def load_policy(path):
    layers = [(np.array(L["W"]), np.array(L["b"])) for L in json.load(open(path))["layers"]]

    def f(o):
        h = o
        for W, b in layers:
            h = np.tanh(W @ h + b)
        return h
    return f


def score(policy, side=1, n=40, seed0=TEST_SEED0, max_steps=2400, world_kw=None, obs_fn=None):
    tot = {"done": 0, "collision": 0, "offroad": 0, "red_runs": 0, "steps": 0, "out_of_lane": 0, "speeding": 0,
           "wrong_signal": 0, "time": 0.0}
    for ep in range(n):
        w = World(seed0 + ep, side=side, **(world_kw or {}))
        ego = w.cars[0]
        for _ in range(max_steps):
            o = w.observe(0)
            act = w.expert(0) if policy is None else policy(obs_fn(o, w) if obs_fn else o)
            e, _, seg = w.reference(ego)
            if not w.in_box(ego.pos, ego.b) and not w.in_box(ego.pos, ego.a):
                tot["out_of_lane"] += abs(e - side * LANE) > 1.2
            tot["speeding"] += ego.v > w.limit[seg] + 1.0
            should = (-1 if ego.turn == 0 else 1) if ego.turn != 1 and (w.stop_dist(ego) < 30 or w.in_box(ego.pos, ego.b)) else 0
            given = 0 if abs(act[2]) < 0.5 else int(np.sign(act[2]))
            tot["wrong_signal"] += given != should
            tot["steps"] += 1
            ev = w.step(act)
            if ev["collision"] or ev["offroad"] or ev["done"]:
                break
        for k in ("done", "collision", "offroad"):
            tot[k] += ev[k]
        tot["red_runs"] += ev["red_runs"]
        tot["time"] += w.t
    s = tot["steps"]
    return {"routes_done": tot["done"] / n, "collision": tot["collision"] / n, "offroad": tot["offroad"] / n,
            "red_runs_per_route": tot["red_runs"] / n, "out_of_lane": tot["out_of_lane"] / s,
            "speeding": tot["speeding"] / s, "wrong_signal": tot["wrong_signal"] / s, "mean_time_s": tot["time"] / n}


def show(name, r):
    print(f"{name:28} done {r['routes_done']:5.0%}  crash {r['collision']:4.0%}  offroad {r['offroad']:4.0%}  "
          f"red/route {r['red_runs_per_route']:.2f}  out-of-lane {r['out_of_lane']:5.1%}  speeding {r['speeding']:5.1%}  "
          f"wrong-signal {r['wrong_signal']:5.1%}  time {r['mean_time_s']:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("driver")
    ap.add_argument("--side", type=int, default=1)
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    pol = None if a.driver == "expert" else load_policy(a.driver)
    show(f"{a.driver} side={a.side}", score(pol, a.side, a.n))
