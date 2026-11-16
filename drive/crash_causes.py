"""What each driver crashes into, and when (60 test routes)."""
import sys
from collections import Counter
from multiprocessing import Pool
import numpy as np
from city import World, CAR_R
from evaluate import load_policy, TEST_SEED0


def run(job):
    path, side, seed = job
    pol, w = load_policy(path), World(seed, side=side)
    for _ in range(4000):
        ev = w.step(pol(w.observe(0)))
        if ev["collision"] or ev["offroad"] or ev["done"]:
            break
    if not ev["collision"]:
        return None
    e = w.cars[0]
    dc = min(np.linalg.norm(e.circles()[:, None] - o.circles()[None], axis=2).min() for o in w.cars[1:])
    dp = min([np.linalg.norm(e.circles() - p[0], axis=1).min() for p in w.peds] + [99])
    what = "pedestrian" if dp < dc else "car"
    where = "in junction" if w.in_box(e.pos, e.b) else "on road"
    return f"{what}, {where}, speed {'<2' if e.v < 2 else '2-6' if e.v < 6 else '>6'} m/s"


if __name__ == "__main__":
    for name, path, side in [("v4 (right)", "nets/driver_v4.json", 1)]:
        with Pool(6) as p:
            res = [r for r in p.map(run, [(path, side, s) for s in range(TEST_SEED0, TEST_SEED0 + 60)]) if r]
        print(name, len(res), "crashes:", dict(Counter(res)))
